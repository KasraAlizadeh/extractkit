"""OpenAI client with structured outputs and retries.

This module is the only place in the package that talks to OpenAI. It
exposes a single :class:`LLMClient` with two methods that mirror the
two-pass extraction strategy:

* :meth:`LLMClient.extract_structured` — pass 1: pull the structured
  factual fields with JSON-schema validation, so the response is
  guaranteed to match :class:`extractkit.schemas.StructuredFields`.
* :meth:`LLMClient.extract_synthesis` — pass 2: ask the model to write
  the summary fields, again returned as a validated Pydantic model.

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
from extractkit.schemas import StructuredFields, SynthesisFields

# Generic bound to Pydantic models so the private helpers can promise
# "you get back exactly the schema you passed in" rather than a union.
SchemaT = TypeVar("SchemaT", bound=BaseModel)


# Domain-aware system prompt that defines the extraction task in the
# user's own words and supplies the vocabulary the model needs to
# recognise indirect mentions of clo, met, PMV, UTCI, Köppen-Geiger
# climates, and the four comfort subjects.
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
    "both the value and its unit when stated (e.g. '0.6 clo', '1.2 met', "
    "'Ta = 28°C', 'PMV from -0.5 to +1.0').\n"
    "4. For location fields (Country, City, Climate), look in the methods "
    "section, study-area description, and figure captions. Climate may be "
    "stated as a Köppen-Geiger code or descriptively — record what is given.\n"
    "5. For Subjects of Study, ONLY use these four labels: 'Visual Comfort', "
    "'Acoustic Comfort', 'Outdoor Air Quality', 'Thermal Comfort'. Choose "
    "all that apply, comma-separated. This describes the research topic, "
    "NOT the participants.\n"
    "6. For author/year/journal/DOI, check the title page, header, footer, "
    "and the reference itself.\n"
    "7. KPIs are calculated comfort indices (PMV, PPD, UTCI, SET*, PET, "
    "WBGT). They are calculation outputs, not raw measurements like Ta.\n"
    "8. When listing multiple items (authors, keywords, variables, KPIs), "
    "separate with commas. Do not use 'and' or bullet points.\n"
    "9. Preserve original spellings of names, places, and technical terms.\n"
    "10. Return EXACTLY the string 'N/A' (no quotes, no other variant like "
    "'not available' or 'unknown') when the article does not mention a field "
    "anywhere. Do not guess or fabricate.\n\n"
    "DOMAIN VOCABULARY YOU SHOULD RECOGNIZE\n"
    "- Comfort indices / KPIs: PMV, PPD, UTCI, SET*, PET, WBGT, OUT_SET*, "
    "TSV (Thermal Sensation Vote), TCV (Thermal Comfort Vote).\n"
    "- Clothing: clo value, Icl, clothing insulation, garment ensemble.\n"
    "- Activity / metabolic rate: met value, M, activity level.\n"
    "- Microclimate variables: Ta (air temperature), Tmrt (mean radiant "
    "temperature), RH (relative humidity), Va (air velocity), Tg (globe "
    "temperature), SR (solar radiation).\n"
    "- Climate zones: Köppen-Geiger codes (Af, Am, Aw, BWh, BWk, BSh, BSk, "
    "Cfa, Cfb, Csa, Csb, Cwa, Dfa, Dfb, etc.).\n"
    "- Study designs: field study, climate chamber, laboratory, questionnaire "
    "survey, simulation (ENVI-met, RayMan, SOLWEIG, CFD), longitudinal, "
    "cross-sectional, transversal.\n"
    "- Urban cooling: street trees, green roofs, cool pavements, water "
    "features, urban geometry, high-albedo surfaces, shading devices.\n"
    "- Common software: ENVI-met, RayMan, SOLWEIG, Ladybug, ANSYS Fluent, "
    "OpenFOAM, SPSS, R, MATLAB, Python."
)


_STRUCTURED_USER_PROMPT: Final[str] = (
    "Extract the requested factual fields from the academic article below.\n\n"
    "INSTRUCTIONS\n"
    "- Search the WHOLE article for each field, not just one section.\n"
    "- Extract values even when phrased indirectly (see the system prompt).\n"
    "- For numerical values, include units (e.g. '0.5 clo', '25°C').\n"
    "- For lists, separate items with commas.\n"
    "- For the 'Subjects of Study' field, use ONLY these labels (choose all "
    "that apply): Visual Comfort, Acoustic Comfort, Outdoor Air Quality, "
    "Thermal Comfort.\n"
    "- Return exactly 'N/A' when a field is not mentioned in the article.\n\n"
    "ARTICLE TEXT\n"
    "---\n"
    "{article_text}\n"
    "---"
)


_SYNTHESIS_USER_PROMPT: Final[str] = (
    "Read the academic article below and produce the requested summary "
    "fields. Each summary should be ONE concise paragraph (3-6 sentences), "
    "written in clear scholarly prose and faithful to what the article "
    "actually says.\n\n"
    "INSTRUCTIONS\n"
    "- Base every claim on content explicitly present in the article.\n"
    "- Do not invent results, statistics, or conclusions.\n"
    "- 'Research Questions' — what the authors set out to answer.\n"
    "- 'Key Goals' — what the authors aimed to achieve.\n"
    "- 'Methodology' — study design, participants, instruments, and "
    "analysis approach in one paragraph.\n"
    "- 'Notes' — limitations, caveats, sample restrictions, future work.\n"
    "- 'Brief Double Click to See All' — one-paragraph executive summary "
    "covering scope, approach, and headline findings.\n"
    "- 'G-M-R Brief' — a single flowing paragraph stating (a) the GOALS, "
    "(b) the METHODOLOGY, and (c) the RESULTS, in that order.\n"
    "- Return exactly 'N/A' when a field cannot be derived from the article.\n\n"
    "ARTICLE TEXT\n"
    "---\n"
    "{article_text}\n"
    "---"
)


# A rough character budget for what we send as the article body. The
# OpenAI models we target (gpt-4o-mini and gpt-4o) accept ~128k tokens,
# which is roughly 500k characters; we stay well under that to leave
# room for the prompt itself and the response. Truncating from the
# middle preserves the abstract/intro (start) and the conclusion (end),
# which is where most of the extractable fields live.
_MAX_ARTICLE_CHARS: Final[int] = 200_000


def _truncate_middle(text: str, limit: int = _MAX_ARTICLE_CHARS) -> str:
    """Trim a long article by removing the middle.

    Academic articles put the most extractable information at the start
    (title, abstract, methods) and at the end (conclusions). The middle
    (results tables, lengthy discussion) is heavier on detail than on
    summary, so dropping it is the least lossy way to fit a long paper
    in the context window.
    """
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n\n[... middle of article truncated ...]\n\n" + text[-half:]


class LLMClient:
    """Thin wrapper around the OpenAI client for extraction.

    The client is constructed once per run and reused across all PDFs,
    so the underlying HTTP connection pool is shared.
    """

    def __init__(self, settings: Settings) -> None:
        """Build the client.

        Args:
            settings: Validated application settings. The API key must
                already be present (call :meth:`Settings.validate_ready`
                first if you want a friendly error).
        """
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.request_timeout,
            max_retries=0,  # we handle retries ourselves via tenacity
        )

    def extract_structured(self, article_text: str) -> StructuredFields:
        """Run pass 1: extract the structured factual fields.

        Args:
            article_text: Full text of the article (any length; long
                articles are truncated transparently).

        Returns:
            A populated :class:`StructuredFields` instance. Fields the
            article does not mention come back as the string 'N/A'.

        Raises:
            LLMError: If the API call fails after all retries, or if the
                response cannot be parsed.
        """
        return self._call_with_schema(
            article_text=article_text,
            user_prompt_template=_STRUCTURED_USER_PROMPT,
            schema_model=StructuredFields,
        )

    def extract_synthesis(self, article_text: str) -> SynthesisFields:
        """Run pass 2: produce the summary fields."""
        return self._call_with_schema(
            article_text=article_text,
            user_prompt_template=_SYNTHESIS_USER_PROMPT,
            schema_model=SynthesisFields,
        )

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
        """Call the OpenAI parse helper, retrying transient failures.

        This is the only place where the raw API call lives, kept
        small so the retry policy is easy to read.
        """
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

    def _call_with_schema(
        self,
        *,
        article_text: str,
        user_prompt_template: str,
        schema_model: type[SchemaT],
    ) -> SchemaT:
        """Shared body for both extraction passes."""
        trimmed = _truncate_middle(article_text)
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": _SYSTEM_PROMPT_BASE},
            {
                "role": "user",
                "content": user_prompt_template.format(article_text=trimmed),
            },
        ]

        try:
            return self._raw_parse(messages=messages, schema_model=schema_model)
        except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
            # Tenacity reraised after exhausting retries; surface as LLMError.
            raise LLMError(f"OpenAI API unavailable after retries: {exc}") from exc
        except OpenAIAPIError as exc:
            # Permanent errors (bad key, invalid request, etc.) — do not retry.
            raise LLMError(f"OpenAI API error: {exc}") from exc
