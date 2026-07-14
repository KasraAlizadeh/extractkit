"""Tests for the extraction schema definitions."""

from __future__ import annotations

from extractkit.schemas import (
    COLUMN_TO_FIELD,
    EXCEL_COLUMNS,
    ArticleExtraction,
    StructuredFields,
    SynthesisFields,
    extraction_to_row,
)


def test_excel_columns_has_expected_count() -> None:
    assert len(EXCEL_COLUMNS) == 39


def test_column_to_field_has_expected_count() -> None:
    assert len(COLUMN_TO_FIELD) == 39


def test_every_excel_column_has_a_field_mapping() -> None:
    for column in EXCEL_COLUMNS:
        assert column in COLUMN_TO_FIELD


def test_field_mappings_target_real_model_fields() -> None:
    structured_fields = set(StructuredFields.model_fields.keys())
    synthesis_fields = set(SynthesisFields.model_fields.keys())
    all_fields = structured_fields | synthesis_fields
    for field_name in COLUMN_TO_FIELD.values():
        assert field_name in all_fields, f"unknown field '{field_name}'"


def test_extraction_to_row_alignment(
    structured_extraction_dict: dict[str, str],
    synthesis_extraction_dict: dict[str, str],
) -> None:
    extraction = ArticleExtraction(
        structured=StructuredFields(**structured_extraction_dict),
        synthesis=SynthesisFields(**synthesis_extraction_dict),
    )
    row = extraction_to_row(extraction)

    assert len(row) == len(EXCEL_COLUMNS)
    combined = {**structured_extraction_dict, **synthesis_extraction_dict}
    for column, value in zip(EXCEL_COLUMNS, row, strict=True):
        field_name = COLUMN_TO_FIELD[column]
        assert value == combined[field_name]


def test_missing_fields_default_to_not_available() -> None:
    structured = StructuredFields()
    assert structured.title == "NA"
    assert structured.year == "NA"
    assert structured.article_classification == "NA"
