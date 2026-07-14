"""Tests for the OpenAI client wrapper."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from openai import APIConnectionError, APITimeoutError, RateLimitError

from extractkit.config import Settings
from extractkit.exceptions import LLMError
from extractkit.llm_client import LLMClient
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


def _fake_response(parsed_object: Any) -> MagicMock:
    message = MagicMock()
    message.parsed = parsed_object
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture
def settings() -> Settings:
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
    d = structured_extraction_dict
    batch_responses = [
        ClassificationFields(
            article_classification=d["article_classification"],
            main_focus=d["main_focus"],
        ),
        BibliographicFields(
            title=d["title"],
            authors=d["authors"],
            year=d["year"],
            journal=d["journal"],
            doi=d["doi"],
        ),
        StudyContextFields(
            country=d["country"],
            city=d["city"],
            climate=d["climate"],
            seasons=d["seasons"],
            urban_space_types=d["urban_space_types"],
        ),
        ParticipantsFields(
            age=d["age"],
            gender=d["gender"],
            ethnicity=d["ethnicity"],
            clothing_level=d["clothing_level"],
            activity_during_exposure=d["activity_during_exposure"],
            activity_before_exposure=d["activity_before_exposure"],
        ),
        MeasurementsFields(
            variables_quantitative=d["variables_quantitative"],
            variables_qualitative=d["variables_qualitative"],
        ),
        SubjectiveComfortFields(
            questionnaire_extent=d["questionnaire_extent"],
            survey_time=d["survey_time"],
            questionnaire_questions=d["questionnaire_questions"],
        ),
        CalculatedIndexesFields(
            calculated_indexes=d["calculated_indexes"],
        ),
        StrategiesFields(
            urban_cooling=d["urban_cooling"],
            personal_cooling=d["personal_cooling"],
            urban_heating=d["urban_heating"],
            personal_heating=d["personal_heating"],
        ),
        ModelingFields(
            modeling_simulation=d["modeling_simulation"],
        ),
        ReviewFields(
            review_scope=d["review_scope"],
            number_of_studies_reviewed=d["number_of_studies_reviewed"],
            themes_categories=d["themes_categories"],
            methods_compared=d["methods_compared"],
            gaps_identified=d["gaps_identified"],
            conclusions=d["conclusions"],
        ),
    ]

    client = LLMClient(settings)
    client._client.beta.chat.completions.parse = MagicMock(  # type: ignore[method-assign]
        side_effect=[_fake_response(r) for r in batch_responses],
    )

    result = client.extract_structured(fake_article_text)

    assert isinstance(result, StructuredFields)
    assert result.title == d["title"]
    assert result.article_classification == d["article_classification"]
    assert result.main_focus == d["main_focus"]
    assert result.doi == d["doi"]
    assert result.calculated_indexes == d["calculated_indexes"]


def test_extract_synthesis_returns_populated_model(
    settings: Settings,
    synthesis_extraction_dict: dict[str, str],
    fake_article_text: str,
) -> None:
    client = LLMClient(settings)
    expected = SynthesisFields(**synthesis_extraction_dict)
    client._client.beta.chat.completions.parse = MagicMock(  # type: ignore[method-assign]
        return_value=_fake_response(expected),
    )

    result = client.extract_synthesis(fake_article_text)

    assert isinstance(result, SynthesisFields)
    assert result.methodology == synthesis_extraction_dict["methodology"]
    assert result.gmr_brief == synthesis_extraction_dict["gmr_brief"]


def test_none_parsed_raises_llm_error(settings: Settings, fake_article_text: str) -> None:
    client = LLMClient(settings)
    client._client.beta.chat.completions.parse = MagicMock(  # type: ignore[method-assign]
        return_value=_fake_response(None),
    )

    with pytest.raises(LLMError):
        client.extract_structured(fake_article_text)


def test_rate_limit_is_retried_and_eventually_raises(
    settings: Settings, fake_article_text: str
) -> None:
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

    assert parse_mock.call_count > 1


def test_timeout_is_retried(settings: Settings, fake_article_text: str) -> None:
    client = LLMClient(settings)
    timeout = APITimeoutError(request=MagicMock())
    parse_mock = MagicMock(side_effect=timeout)
    client._client.beta.chat.completions.parse = parse_mock  # type: ignore[method-assign]

    with pytest.raises(LLMError):
        client.extract_synthesis(fake_article_text)

    assert parse_mock.call_count > 1


def test_connection_error_is_retried(settings: Settings, fake_article_text: str) -> None:
    client = LLMClient(settings)
    conn_err = APIConnectionError(request=MagicMock())
    parse_mock = MagicMock(side_effect=conn_err)
    client._client.beta.chat.completions.parse = parse_mock  # type: ignore[method-assign]

    with pytest.raises(LLMError):
        client.extract_structured(fake_article_text)

    assert parse_mock.call_count > 1
