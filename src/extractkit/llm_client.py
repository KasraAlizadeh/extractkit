"""OpenAI client with structured outputs and retries.

This module is the only place in the package that talks to OpenAI. It
exposes a single :class:`LLMClient` with two methods that mirror the
extraction strategy:

* :meth:`LLMClient.extract_structured` runs SIX focused sub-passes — one
  each for bibliographic data, study context, participants, measurements,
  strategies, and tooling — and merges the results. Splitting the task
  this way gives the model its full attention on each batch and produces
  noticeably more complete extractions than asking for all 28 fields at
  once.
* :meth:`LLMClient.extract_synthesis` runs a single pass to produce the
  six free-form summary fields.

Transient failures (rate limits, timeouts, server errors) are retried
with exponential backoff via :mod:`tenacity`; permanent failures (bad
API key, malformed request) fail fast so the user sees the real cause.
"""

from __future__ import annotations

from typing import Final, TypeVar

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from openai import APIError as OpenAIAPIError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from extractkit.config import Settings
from extractkit.exceptions import LLMError
from extractkit.schemas import (
    BibliographicFields,
    MeasurementsFields,
    ParticipantsFields,
    StrategiesFields,
    StructuredFields,
    StudyContextFields,
    SynthesisFields,
    ToolingFields,
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


_SYSTEM_PROMPT_BASE: Final[str] = (
    "You are an expert research assistant for systematic literature reviews "
    "in environmental and human comfort research, with deep knowledge of "
    "thermal comfort, ASHRAE-55, urban heat island studies, visual comfort, "
    "acoustic comfort, and outdoor air quality.\n\n"
    "TASK\n"
    "Given an academic article, extract the requested fields and produce "
    "the requested summaries with high accuracy.\n\n"
    "EXTRACTION RULES\n"
    "1. Read the WHOLE article carefully — abstract, methods, results, "
    "discussion, conclusion, and figure captions — before extracting any "
    "field. Relevant information is often distributed across sections.\n"
    "2. Extract information even when phrased indirectly. For example, "
    "'participants wore summer attire' implies a clothing level of ~0.5 clo. "
    "'Subjects performed light office work' implies ~1.2 met.\n"
    "3. For numerical fields (clo, met, PMV, UTCI, Ta, RH, etc.), include "
    "both value and unit (e.g. '0.6 clo', 'Ta = 28°C', 'PMV from -0.5 to "
    "+1.0').\n"
    "4. For Subjects of Study, ONLY use these four labels: 'Visual Comfort', "
    "'Acoustic Comfort', 'Outdoor Air Quality', 'Thermal Comfort'. Choose "
    "all that apply, comma-separated.\n"
    "5. KPIs are calculated comfort indices (PMV, PPD, UTCI, SET*, PET, "
    "WBGT) — calculation outputs, not raw measurements like Ta.\n"
    "6. When listing multiple items, separate with commas. Do not use 'and' "
    "or bullet points.\n"
    "7. Preserve original spellings.\n"
    "8. Return exactly the string 'N/A' (no quotes, no other variant) when "
    "a field is not stated anywhere in the article. Do not guess.\n\n"
    "DOMAIN VOCABULARY\n"
    "- Comfort indices / KPIs: PMV, PPD, UTCI, SET*, PET, WBGT, OUT_SET*, "
    "TSV, TCV.\n"
    "- Clothing: clo, Icl, garment ensemble.\n"
    "- Activity: met, M, metabolic rate.\n"
    "- Microclimate variables: Ta, Tmrt, RH, Va, Tg, SR.\n"
    "- Köppen-Geiger climate codes: Af, Am, Aw, BWh, BWk, BSh, BSk, Cfa, "
    "Cfb, Csa, Csb, Cwa, Dfa, Dfb, etc.\n"
    "- Study designs: field study, climate chamber, laboratory, "
    "questionnaire survey, simulation, longitudinal, cross-sectional.\n"
    "- Urban cooling: street trees, green roofs, cool pavements, water "
    "features, urban geometry, high-albedo surfaces.\n"
    "- Common software: ENVI-met, RayMan, SOLWEIG, Ladybug, ANSYS Fluent, "
    "OpenFOAM, SPSS, R, MATLAB, Python."
)


# Per-batch user prompts. Each one names the batch so the model focuses
# its attention. The list-of-fields hint is implicit via the JSON schema
# the SDK sends alongside, so we keep these prompts short.
_BATCH_USER_PROMPTS: Final[dict[str, str]] = {
    "bibliographic": (
        "Extract the BIBLIOGRAPHIC fields from the article below: title, "
        "keywords, year, journal, DOI, and authors. Search the title page, "
        "running header / footer, and reference. Return 'N/A' for any field "
        "not stated."
    ),
    "study_context": (
        "Extract the STUDY CONTEXT fields from the article below: country, "
        "city, climate (prefer Köppen-Geiger code), the comfort domain(s) "
        "studied (Visual / Acoustic / Outdoor Air Quality / Thermal "
        "Comfort), the seasons of study, and the type of urban space. "
        "Return 'N/A' for any field not stated."
    ),
    "participants": (
        "Extract the PARTICIPANT fields from the article below: age, "
        "gender distribution, ethnicity, behaviours / activity level "
        "(prefer met values), and clothing level (prefer clo values). "
        "Return 'N/A' for any field not stated."
    ),
    "measurements": (
        "Extract the MEASUREMENT fields from the article below: "
        "quantitative variables (Ta, RH, Tmrt, Va, etc.), qualitative "
        "variables (TSV, TCV, etc.), questionnaire extent (number of "
        "participants), timing of the questionnaire, summary of "
        "questionnaire topics, and KPIs (PMV, PPD, UTCI, etc.). Return "
        "'N/A' for any field not stated."
    ),
    "strategies": (
        "Extract the CONTROL-STRATEGY fields from the article below: "
        "urban-scale cooling strategies, personal-scale cooling strategies, "
        "urban-scale heating strategies, and personal-scale heating "
        "strategies. Return 'N/A' for any strategy type the article does "
        "not investigate."
    ),
    "tooling": (
        "Extract the SOFTWARE / TOOLING field from the article below: "
        "all software, simulation tools, and statistical packages used "
        "(ENVI-met, RayMan, SOLWEIG, ANSYS Fluent, OpenFOAM, SPSS, R, "
        "MATLAB, Python, etc.). State if a custom / native tool was "
        "developed. Return 'N/A' if no software is mentioned."
    ),
}


_SYNTHESIS_USER_PROMPT: Final[str] = (
    "Read the academic article below and produce the requested summary "
    "fields. Each summary should be ONE concise paragraph (3-6 sentences), "
    "written in clear scholarly prose and faithful to the article.\n\n"
    "- Research Questions — what the authors set out to answer.\n"
    "- Key Goals — what the authors aimed to achieve.\n"
    "- Methodology — study design, participants, instruments, analysis.\n"
    "- Notes — limitations, caveats, future work.\n"
    "- Brief Double Click to See All — executive summary.\n"
    "- G-M-R Brief — single paragraph stating (a) GOALS, (b) METHODOLOGY, "
    "(c) RESULTS, in that order.\n"
    "Return exactly 'N/A' if a summary cannot be derived."
)


# Character budget per API call. 200k chars (~50k tokens) leaves plenty
# of room within gpt-4o-mini's 128k-token window for our prompts and the
# response, while keeping the article body large enough that most papers
# fit in their entirety.
_MAX_ARTICLE_CHARS: Final[int] = 200_000


def _truncate_middle(text: str, limit: int = _MAX_ARTICLE_CHARS) -> str:
    """Trim a long article by removing the middle.

    Academic articles put the most extractable information at the start
    (title, abstract, methods) and end (conclusions). Dropping the middle
    (results tables, lengthy discussion) is the least lossy way to fit a
    long paper in the context window.
    """
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n\n[... middle of article truncated ...]\n\n" + text[-half:]


class LLMClient:
    """Thin wrapper around the OpenAI client for extraction.

    Constructed once per run; the underlying HTTP connection pool is
    reused across every PDF.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.request_timeout,
            max_retries=0,  # we handle retries ourselves via tenacity
        )

    def extract_structured(self, article_text: str) -> StructuredFields:
        """Run all structured batches and merge into one StructuredFields.

        Each batch is a separate API call so the model can focus its
        attention on a small, related group of fields, which produces
        markedly more complete extractions than a single 28-field call.
        """
        trimmed = _truncate_middle(article_text)

        bib = self._batch_call(BibliographicFields, "bibliographic", trimmed)
        ctx = self._batch_call(StudyContextFields, "study_context", trimmed)
        ppl = self._batch_call(ParticipantsFields, "participants", trimmed)
        msr = self._batch_call(MeasurementsFields, "measurements", trimmed)
        stg = self._batch_call(StrategiesFields, "strategies", trimmed)
        tol = self._batch_call(ToolingFields, "tooling", trimmed)

        return StructuredFields(
            **bib.model_dump(),
            **ctx.model_dump(),
            **ppl.model_dump(),
            **msr.model_dump(),
            **stg.model_dump(),
            **tol.model_dump(),
        )

    def extract_synthesis(self, article_text: str) -> SynthesisFields:
        """Run the synthesis pass to produce the six summary fields."""
        trimmed = _truncate_middle(article_text)
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": _SYSTEM_PROMPT_BASE},
            {
                "role": "user",
                "content": (f"{_SYNTHESIS_USER_PROMPT}\n\nARTICLE TEXT\n---\n{trimmed}\n---"),
            },
        ]
        return self._raw_parse_with_handling(
            messages=messages,
            schema_model=SynthesisFields,
        )

    def _batch_call(
        self,
        schema_model: type[SchemaT],
        batch_key: str,
        trimmed_text: str,
    ) -> SchemaT:
        """One focused sub-call for a structured-field batch."""
        user_intro = _BATCH_USER_PROMPTS[batch_key]
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": _SYSTEM_PROMPT_BASE},
            {
                "role": "user",
                "content": (f"{user_intro}\n\nARTICLE TEXT\n---\n{trimmed_text}\n---"),
            },
        ]
        return self._raw_parse_with_handling(
            messages=messages,
            schema_model=schema_model,
        )

    def _raw_parse_with_handling(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        schema_model: type[SchemaT],
    ) -> SchemaT:
        """Call ``_raw_parse`` and translate OpenAI errors to LLMError."""
        try:
            return self._raw_parse(messages=messages, schema_model=schema_model)
        except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
            raise LLMError(f"OpenAI API unavailable after retries: {exc}") from exc
        except OpenAIAPIError as exc:
            raise LLMError(f"OpenAI API error: {exc}") from exc

    @retry(
        retry=retry_if_exception_type(
            (RateLimitError, APITimeoutError, APIConnectionError),
        ),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _raw_parse(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        schema_model: type[SchemaT],
    ) -> SchemaT:
        """Call the OpenAI parse helper, retrying transient failures."""
        response = self._client.beta.chat.completions.parse(
            model=self._settings.model,
            messages=messages,
            response_format=schema_model,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise LLMError(
                "OpenAI returned a response without a parsed object. "
                "This usually means the model refused or produced "
                "invalid JSON."
            )
        return parsed
