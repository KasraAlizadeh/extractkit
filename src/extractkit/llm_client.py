"""OpenAI client with structured outputs and retries.

Runs TEN focused structured batches per article (classification,
bibliographic, study context, participants, measurements, subjective
comfort, indexes, strategies, modeling, review) plus one synthesis
pass for the four GMR fields. Splitting the extraction this way lets
the model focus on a small topic per call and produces markedly more
complete extractions than a single 35-field call.

The system prompt is the strict information-extraction contract from
the extraction spec: extract only what the article explicitly states,
use the article's wording, use 'NA' / 'UNCERTAIN' / 'MULTIPLE' for
absent or ambiguous fields, pipe-separate multi-values.

Transient failures are retried via :mod:`tenacity`; permanent failures
fail fast so the user sees the real cause.
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
    CalculatedIndexesFields,
    ClassificationFields,
    MeasurementsFields,
    ModelingFields,
    ParticipantsFields,
    ReviewFields,
    StrategiesFields,
    StructuredFields,
    StudyContextFields,
    SubjectiveComfortFields,
    SynthesisFields,
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


_SYSTEM_PROMPT_BASE: Final[str] = (
    "You are a scientific literature extraction engine specialized in "
    "outdoor environmental comfort research. Your only task is to read a "
    "scientific article and extract its content into a strict 39-column "
    "database schema. You do not summarize, paraphrase, interpret, or "
    "infer. You only extract what is explicitly stated in the article. "
    "Your output must always contain all requested fields, in order, "
    "with no exceptions.\n\n"
    "ABSOLUTE RULES (apply to every single field)\n"
    "1. Extract only what is explicitly stated in the article text, "
    "tables, captions, or figures.\n"
    "2. Do not infer, guess, generalize, or fill gaps using external "
    "knowledge.\n"
    "3. Do not merge or collapse separate fields into one.\n"
    "4. For every field, use exactly one of the following when content "
    "is absent:\n"
    "   - NA -> not reported, not applicable, or not relevant to this "
    "study type\n"
    "   - UNCERTAIN -> mentioned ambiguously, contradictory, or only "
    "implied\n"
    "   - MULTIPLE -> more than one distinct value applies with no "
    "clear primary\n"
    "5. Preserve the article's exact terminology, units, abbreviations, "
    "and scales.\n"
    "6. When a field contains multiple values, separate them with ' | ' "
    "(pipe character).\n"
    "7. Never write explanatory text, labels, or headers in the cell "
    "values.\n\n"
    "EVIDENCE PRIORITY ORDER\n"
    "Use sources in this order of priority (higher = more reliable):\n"
    "1. Main body text\n"
    "2. Tables\n"
    "3. Captions and footnotes\n"
    "4. Figures\n"
    "5. Abstract and keywords\n"
    "If values conflict across sources, use the higher-priority source "
    "and mark UNCERTAIN if the conflict is significant.\n\n"
    "STUDY-TYPE SENSITIVITY\n"
    "First identify the article type internally:\n"
    "- Field Study\n"
    "- Simulation Study\n"
    "- Review\n"
    "- Mixed (Field + Simulation)\n"
    "- Method / Model Paper\n"
    "- Machine Learning / Predictive Study\n"
    "- Case Study\n"
    "- Other\n\n"
    "Simulation-only studies -> participant fields are NA unless human "
    "subjects were explicitly included.\n"
    "Review articles -> subject-level and survey fields are NA unless "
    "the review aggregates subject-level data.\n"
    "Method / model papers -> fill modeling fields thoroughly; most "
    "subject and survey fields are NA.\n\n"
    "OUTPUT DISCIPLINE\n"
    "- No field is blank — every field has a value, NA, UNCERTAIN, or "
    "MULTIPLE.\n"
    "- No field contains explanatory text about why data is absent — "
    "just write NA.\n"
    "- Pipe character ' | ' is used consistently for all multi-value "
    "cells.\n"
    "- Units are preserved wherever stated in the article.\n"
    "- Do not add commentary, explanations, headings, or notes outside "
    "the schema."
)


_BATCH_USER_PROMPTS: Final[dict[str, str]] = {
    "classification": (
        "Determine the ARTICLE CLASSIFICATION and MAIN FOCUS for the "
        "article below. Article Classification is a single best match "
        "from: Field Study | Simulation Study | Review | Mixed (Field + "
        "Simulation) | Method / Model Paper | Machine Learning / "
        "Predictive Study | Case Study | Other (specify). Main Focus "
        "lists all comfort focuses that apply from: Thermal Comfort | "
        "Outdoor Air Quality | Visual Comfort | Acoustic Comfort | Multi-"
        "Sensory Comfort | Calibration / Assessment Tools | Other "
        "(specify), separated by ' | '. Apply the strict rules from the "
        "system prompt."
    ),
    "bibliographic": (
        "Extract the BIBLIOGRAPHIC identifiers for the article below: "
        "full title (no truncation), all authors (Last, First | Last, "
        "First; first 6 then 'et al.' if more), 4-digit year, full "
        "journal or conference name, and DOI with 'https://doi.org/' "
        "prefix (or ISBN, or other identifier). Apply the strict rules "
        "from the system prompt."
    ),
    "study_context": (
        "Extract the STUDY-CONTEXT fields for the article below: "
        "country, city, climate (exact description stated), season(s) or "
        "months studied, and urban space types (article's own "
        "terminology; mark simulation-only spaces as '(simulated)'). "
        "Apply the strict rules from the system prompt."
    ),
    "participants": (
        "Extract the PARTICIPANT / human-subject fields for the article "
        "below: age (ranges, means, or group labels), gender "
        "distribution, ethnicity / nationality / participant group, "
        "clothing level (description AND clo values if given), activity "
        "during exposure, and activity up to 24 hours before exposure "
        "(only if explicitly stated). If the article has no human "
        "subjects (simulation-only, model-only, or review-only), every "
        "field is 'NA'. Assumed simulation values are noted "
        "'(assumed)'. Apply the strict rules from the system prompt."
    ),
    "measurements": (
        "Extract the ENVIRONMENTAL MEASUREMENT variables for the article "
        "below: quantitative variables (with units and article "
        "abbreviations, e.g. 'Ta (°C) | RH (%) | Tmrt (°C) | PET (°C)') "
        "and qualitative variables (subjective vote types with scales, "
        "e.g. 'TSV (ASHRAE 7-point scale: -3 to +3)'). Apply the strict "
        "rules from the system prompt."
    ),
    "subjective_comfort": (
        "Extract the SUBJECTIVE COMFORT survey information for the "
        "article below: questionnaire extent (distributed | valid | "
        "response rate; sessions / locations / rounds), survey time "
        "(exact time windows, hours, temporal protocols, seasonal "
        "context), and questionnaire questions (types / categories, or "
        "verbatim if quoted). If no survey was conducted, all three "
        "fields are 'NA'. Apply the strict rules from the system prompt."
    ),
    "indexes": (
        "Extract the CALCULATED COMFORT INDEXES / KPIs for the article "
        "below. For each index: name | equation / formula if stated | "
        "thresholds, neutral ranges, calibrated values if reported. "
        "Separate multiple indexes with ' | '. If no index is "
        "calculated, return 'NA'. Apply the strict rules from the system "
        "prompt."
    ),
    "strategies": (
        "Extract the FOUR distinct CONTROL-STRATEGY fields for the "
        "article below (do NOT merge them): urban cooling strategies "
        "(urban-scale summer interventions), personal cooling strategies "
        "(individual-level summer behaviours), urban heating strategies "
        "(urban-scale winter interventions), and personal heating "
        "strategies (individual-level winter behaviours). Only include "
        "strategies explicitly studied, reported, observed, or "
        "recommended. Return 'NA' for any category not present. Apply "
        "the strict rules from the system prompt."
    ),
    "modeling": (
        "Extract the MODELING / SIMULATION information for the article "
        "below, combined into ONE cell with labelled sub-elements: "
        "SOFTWARE (name and version), PURPOSE, INPUTS, OUTPUTS, "
        "CALIBRATION / VALIDATION, STATISTICAL / ML METHODS. Format as "
        "pipe-separated flowing text with labels. If the article "
        "involves no computational modeling or simulation, return 'NA'. "
        "Apply the strict rules from the system prompt."
    ),
    "review": (
        "Extract the REVIEW-STUDY fields for the article below (only if "
        "this article is a review or meta-analysis): review scope "
        "(topical, geographic, temporal), number of studies reviewed, "
        "themes / categories, methods compared, gaps identified, and "
        "conclusions. If this article is NOT a review, every field is "
        "'NA'. Apply the strict rules from the system prompt."
    ),
}


_SYNTHESIS_USER_PROMPT: Final[str] = (
    "Read the article below and extract the four GMR fields: Key "
    "Research Questions, Key Goals, Methodology, and GMR Brief.\n\n"
    "- Research Questions: explicit questions stated, or implied "
    "questions from stated objectives. Multiple separated by ' | '.\n"
    "- Key Goals: explicit aims / objectives in the article's own "
    "language. Multiple separated by ' | '.\n"
    "- Methodology: 2-5 sentences on design, instruments, sample "
    "strategy, analysis. Synthesize from methods section — do not copy "
    "the abstract.\n"
    "- GMR Brief: ONE sentence, MAX 40 words, format '[Goal] using "
    "[method], finding that [result].' No citations, no extra context.\n\n"
    "Apply the strict rules from the system prompt (NA / UNCERTAIN / "
    "MULTIPLE labels, no invention, article's own wording)."
)


_MAX_ARTICLE_CHARS: Final[int] = 200_000


def _truncate_middle(text: str, limit: int = _MAX_ARTICLE_CHARS) -> str:
    """Trim a long article by removing the middle.

    Academic articles put the most extractable information at the start
    and end. Dropping the middle is the least lossy way to fit a long
    paper in the context window.
    """
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n\n[... middle of article truncated ...]\n\n" + text[-half:]


class LLMClient:
    """Thin wrapper around the OpenAI client for extraction."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.request_timeout,
            max_retries=0,
        )

    def extract_structured(self, article_text: str) -> StructuredFields:
        """Run all structured batches and merge into one StructuredFields."""
        trimmed = _truncate_middle(article_text)

        cls = self._batch_call(ClassificationFields, "classification", trimmed)
        bib = self._batch_call(BibliographicFields, "bibliographic", trimmed)
        ctx = self._batch_call(StudyContextFields, "study_context", trimmed)
        ppl = self._batch_call(ParticipantsFields, "participants", trimmed)
        msr = self._batch_call(MeasurementsFields, "measurements", trimmed)
        sub = self._batch_call(SubjectiveComfortFields, "subjective_comfort", trimmed)
        idx = self._batch_call(CalculatedIndexesFields, "indexes", trimmed)
        stg = self._batch_call(StrategiesFields, "strategies", trimmed)
        mdl = self._batch_call(ModelingFields, "modeling", trimmed)
        rev = self._batch_call(ReviewFields, "review", trimmed)

        return StructuredFields(
            **cls.model_dump(),
            **bib.model_dump(),
            **ctx.model_dump(),
            **ppl.model_dump(),
            **msr.model_dump(),
            **sub.model_dump(),
            **idx.model_dump(),
            **stg.model_dump(),
            **mdl.model_dump(),
            **rev.model_dump(),
        )

    def extract_synthesis(self, article_text: str) -> SynthesisFields:
        """Run the synthesis pass to produce the four GMR summary fields."""
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
