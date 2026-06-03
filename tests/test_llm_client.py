"""Tests for the OpenAI client wrapper.

Mocking strategy: we patch the OpenAI client at the SDK boundary so the
real network call is never made. The fakes return objects shaped like
what ``openai.beta.chat.completions.parse`` returns at runtime, just
enough for ``LLMClient`` to be exercised end-to-end.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from openai import APIConnectionError, APITimeoutError, RateLimitError

from extractkit.config import Settings
from extractkit.exceptions import LLMError
from extractkit.llm_client import LLMClient
from extractkit.schemas import StructuredFields, SynthesisFields


def _fake_response(parsed_object: Any) -> MagicMock:
    """Build a stand-in for ``client.beta.chat.completions.parse``'s return.

    The real return type is a ``ParsedChatCompletion``; for our purposes
    only ``.choices[0].message.parsed`` matters, so the fake exposes
    just that path.
    """
    message = MagicMock()
    message.parsed = parsed_object
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture
def settings() -> Settings:
    """Minimal valid settings with a placeholder API key."""
    return Settings(
        OPENAI_API_KEY="sk-test-not-real",
        model="gpt-4o-mini",
        request_timeout=5.0,
        max_retries=2,
    )


def test_extract_structured_returns_populated_model(
    settings: Settings,
    structured_extraction_dict: dict[str, str],
    fake_article_text: str,
) -> None:
    """A successful structured-pass call returns a parsed Pydantic model."""
    client = LLMClient(settings)
    expected = StructuredFields(**structured_extraction_dict)
    client._client.beta.chat.completions.parse = MagicMock(  # type: ignore[method-assign]
        return_value=_fake_response(expected),
    )

    result = client.extract_structured(fake_article_text)

    assert isinstance(result, StructuredFields)
    assert result.article_name == structured_extraction_dict["article_name"]
    assert result.year == structured_extraction_dict["year"]


def test_extract_synthesis_returns_populated_model(
    settings: Settings,
    synthesis_extraction_dict: dict[str, str],
    fake_article_text: str,
) -> None:
    """A successful synthesis-pass call returns a parsed Pydantic model."""
    client = LLMClient(settings)
    expected = SynthesisFields(**synthesis_extraction_dict)
    client._client.beta.chat.completions.parse = MagicMock(  # type: ignore[method-assign]
        return_value=_fake_response(expected),
    )

    result = client.extract_synthesis(fake_article_text)

    assert isinstance(result, SynthesisFields)
    assert result.methodology == synthesis_extraction_dict["methodology"]


def test_none_parsed_raises_llm_error(settings: Settings, fake_article_text: str) -> None:
    """If OpenAI returns no parsed object, surface a clear LLMError."""
    client = LLMClient(settings)
    client._client.beta.chat.completions.parse = MagicMock(  # type: ignore[method-assign]
        return_value=_fake_response(None),
    )

    with pytest.raises(LLMError):
        client.extract_structured(fake_article_text)


def test_rate_limit_is_retried_and_eventually_raises(
    settings: Settings, fake_article_text: str
) -> None:
    """Retries happen, but a persistent rate limit still surfaces as LLMError."""
    client = LLMClient(settings)
    rate_limit = RateLimitError(
        message="rate limited",
        response=MagicMock(),
        body=None,
    )
    parse_mock = MagicMock(side_effect=rate_limit)
    client._client.beta.chat.completions.parse = parse_mock  # type: ignore[method-assign]

    with pytest.raises(LLMError):
        client.extract_structured(fake_article_text)

    # Tenacity is configured for up to 5 attempts; we just confirm it
    # was called more than once, proving retries happened.
    assert parse_mock.call_count > 1


def test_timeout_is_retried(settings: Settings, fake_article_text: str) -> None:
    """Timeouts are transient and should trigger retries."""
    client = LLMClient(settings)
    timeout = APITimeoutError(request=MagicMock())
    parse_mock = MagicMock(side_effect=timeout)
    client._client.beta.chat.completions.parse = parse_mock  # type: ignore[method-assign]

    with pytest.raises(LLMError):
        client.extract_synthesis(fake_article_text)

    assert parse_mock.call_count > 1


def test_connection_error_is_retried(settings: Settings, fake_article_text: str) -> None:
    """Network errors are transient and should trigger retries."""
    client = LLMClient(settings)
    conn_err = APIConnectionError(request=MagicMock())
    parse_mock = MagicMock(side_effect=conn_err)
    client._client.beta.chat.completions.parse = parse_mock  # type: ignore[method-assign]

    with pytest.raises(LLMError):
        client.extract_structured(fake_article_text)

    assert parse_mock.call_count > 1
