"""Shared pytest fixtures.

Anything defined here is automatically available in every test in the
``tests/`` package without explicit import. We keep fixtures here for
two reasons: tests stay short and focused, and the cost of building a
fixture (e.g. writing a workbook to disk) is paid once per scope.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from openpyxl import Workbook

from extractkit.schemas import EXCEL_COLUMNS


@pytest.fixture
def template_xlsx(tmp_path: Path) -> Path:
    """Create a minimal Excel template with the canonical 33 headers.

    Yields the path to a freshly created workbook that lives only for
    the duration of the test (``tmp_path`` is cleaned up automatically).
    """
    path = tmp_path / "template.xlsx"
    workbook = Workbook()
    worksheet = workbook.active
    assert worksheet is not None  # narrows the optional type for mypy
    worksheet.append(list(EXCEL_COLUMNS))
    workbook.save(path)
    return path


@pytest.fixture
def sample_row() -> list[str]:
    """A plausible row of extracted values aligned to ``EXCEL_COLUMNS``.

    Real LLM output will have varied content; these placeholders just
    let us verify the writer puts values in the right columns.
    """
    return [f"value-{i}" for i in range(len(EXCEL_COLUMNS))]


@pytest.fixture
def fake_article_text() -> str:
    """A short stand-in for a real article body.

    Long enough to exercise truncation guards without bloating tests.
    """
    return (
        "Title: A Study of Outdoor Thermal Comfort\n"
        "Authors: Smith, Lee\n"
        "Year: 2023\n"
        "Methods: We administered a survey to 120 office workers...\n"
        "Conclusion: Personal cooling strategies were most effective."
    )


@pytest.fixture
def structured_extraction_dict() -> dict[str, Any]:
    """Plausible structured-pass output as plain Python data.

    Used by the LLM-mock tests to simulate what OpenAI's parsed response
    would look like for ``StructuredFields``.
    """
    return {
        "article_name": "A Study of Outdoor Thermal Comfort",
        "key_words": "thermal comfort, outdoor, survey",
        "year": "2023",
        "journal": "Building and Environment",
        "researchers": "Smith, Lee",
        "country": "USA",
        "city": "Phoenix",
        "climate": "BWh",
        "subjects_of_study": "120 office workers",
        "seasons_of_study": "summer",
        "spaces_studied": "outdoor plaza",
        "age": "25-55",
        "gender": "60% male",
        "ethnicity": "",
        "behaviours_activity": "1.2 met",
        "clothing_level": "0.5 clo",
        "numerical_variables": "Ta, RH, Tmrt",
        "qualitative_variables": "thermal sensation votes",
        "questionnaire_extent": "20 items",
        "questionning_time": "every 15 minutes",
        "questionnaire_questions": "ASHRAE-55 7-point scale",
        "kpi": "PMV, UTCI",
        "urban_cooling_strategies": "tree canopy",
        "personal_cooling_strategies": "handheld fans",
        "urban_heating_strategies": "",
        "personal_heating_strategies": "",
        "software_used": "RayMan",
    }


@pytest.fixture
def synthesis_extraction_dict() -> dict[str, Any]:
    """Plausible synthesis-pass output as plain Python data."""
    return {
        "research_questions": "How effective are urban cooling strategies?",
        "key_goals": "Quantify thermal comfort improvement.",
        "methodology": "Field survey combined with microclimate measurements.",
        "notes": "Sample limited to one city.",
        "brief_double_click_to_see_all": "Outdoor thermal comfort study in Phoenix.",
        "g_m_r_brief": (
            "Goal: assess outdoor thermal comfort. Method: 120-participant "
            "field study. Result: tree canopy and handheld fans both "
            "reduced thermal stress."
        ),
    }
