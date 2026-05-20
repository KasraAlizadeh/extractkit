"""OpenAI client with structured outputs and retries.

This module is the only place in the package that talks to OpenAI. It
exposes a single :class:`LLMClient` with two methods that mirror the
two-pass extraction strategy:

* :meth:`LLMClient.extract_structured` — pass 1: pull the 27 factual
  fields with JSON-schema validation, so the response is guaranteed to
  match :class:`extractkit.schemas.StructuredFields`.
* :meth:`LLMClient.extract_synthesis` — pass 2: ask the model to write
  the 6 summary fields, again returned as a validated Pydantic model.

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


# A short, domain-aware system prompt steers the model toward the
# vocabulary used in thermal-comfort research (clo/met, PMV/PPD, UTCI,
# Köppen-Geiger, etc.). Without this, generic prompts tend to produce
# vaguer answers for specialised fields like "Clothing Level".
_SYSTEM_PROMPT_BASE: Final[str] = (
    "You are an expert research assistant specialising in thermal comfort, "
    "ASHRAE-55, and urban heat island studies. You extract information "
    "from academic articles precisely and conservatively. "
    "When a field is not stated in the article, return an empty string "
    "rather than guessing. Use the vocabulary of the field where it "
    "applies (clo values for clothing, met values for activity, PMV/PPD "
    "and UTCI for thermal indices, Köppen-Geiger codes for climate)."
)


_STRUCTURED_USER_PROMPT: Final[str] = (
    "Extract the following factual fields from the academic article below. "
    "Return only what the article explicitly states. Leave fields empty "
    "when the article does not provide the information.\n\n"
    "Article text:\n"
    "---\n"
    "{article_text}\n"
    "---"
)


_SYNTHESIS_USER_PROMPT: Final[str] = (
    "Read the academic article below and produce the requested summary "
    "fields. Each summary should be one paragraph, concise and faithful "
    "to the article. Do not invent results that are not in the text.\n\n"
    "Article text:\n"
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
        """Run pass 1: extract the 27 factual fields.

        Args:
            article_text: Full text of the article (any length; long
                articles are truncated transparently).

        Returns:
            A populated :class:`StructuredFields` instance. Fields the
            article does not mention come back as empty strings.

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
        """Run pass 2: produce the 6 summary fields."""
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
