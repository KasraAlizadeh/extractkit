"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from openpyxl import Workbook

from extractkit.schemas import EXCEL_COLUMNS


@pytest.fixture
def template_xlsx(tmp_path: Path) -> Path:
    """Create a minimal Excel template with the canonical headers."""
    path = tmp_path / "template.xlsx"
    workbook = Workbook()
    worksheet = workbook.active
    assert worksheet is not None
    worksheet.append(list(EXCEL_COLUMNS))
    workbook.save(path)
    return path


@pytest.fixture
def sample_row() -> list[str]:
    """A plausible row of extracted values aligned to ``EXCEL_COLUMNS``."""
    return [f"value-{i}" for i in range(len(EXCEL_COLUMNS))]


@pytest.fixture
def fake_article_text() -> str:
    """A short stand-in for a real article body."""
    return (
        "Title: A Study of Outdoor Thermal Comfort\n"
        "Authors: Smith, Lee\n"
        "Year: 2023\n"
        "Methods: We administered a survey to 120 office workers...\n"
        "Conclusion: Personal cooling strategies were most effective."
    )


@pytest.fixture
def structured_extraction_dict() -> dict[str, Any]:
    """Plausible structured-pass output as plain Python data."""
    return {
        "article_classification": "Field Study",
        "main_focus": "Thermal Comfort",
        "title": "A Study of Outdoor Thermal Comfort",
        "authors": "Smith, John | Lee, Kim",
        "year": "2023",
        "journal": "Building and Environment",
        "doi": "https://doi.org/10.1016/j.buildenv.2023.example",
        "country": "USA",
        "city": "Phoenix",
        "climate": "BWh (hot arid)",
        "seasons": "Summer (July-August)",
        "urban_space_types": "outdoor plaza | street canyon",
        "age": "18-55 years (mean 34)",
        "gender": "60% male, 40% female",
        "ethnicity": "NA",
        "clothing_level": "Light summer clothing (0.5 clo)",
        "activity_during_exposure": "Standing at survey points",
        "activity_before_exposure": "NA",
        "variables_quantitative": "Ta (°C) | RH (%) | Tmrt (°C)",
        "variables_qualitative": "TSV (ASHRAE 7-point scale: -3 to +3)",
        "questionnaire_extent": "120 valid responses",
        "survey_time": "09:00-17:00 daily",
        "questionnaire_questions": "ASHRAE thermal sensation",
        "calculated_indexes": "PMV | UTCI",
        "urban_cooling": "tree canopy",
        "personal_cooling": "handheld fans",
        "urban_heating": "NA",
        "personal_heating": "NA",
        "modeling_simulation": "NA",
        "review_scope": "NA",
        "number_of_studies_reviewed": "NA",
        "themes_categories": "NA",
        "methods_compared": "NA",
        "gaps_identified": "NA",
        "conclusions": "NA",
    }


@pytest.fixture
def synthesis_extraction_dict() -> dict[str, Any]:
    """Plausible synthesis-pass output as plain Python data."""
    return {
        "research_questions": "How effective are urban cooling strategies?",
        "key_goals": "Quantify thermal comfort improvement.",
        "methodology": "Field survey combined with microclimate measurements.",
        "gmr_brief": (
            "Evaluate outdoor thermal comfort in Phoenix using a 120-"
            "participant field survey, finding that tree canopies "
            "reduced PET by up to 5°C."
        ),
    }
