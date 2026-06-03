"""Integration tests for the orchestrator.

These tests exercise the full pipeline — pdf_reader → llm_client →
excel_handler → checkpoint — with the LLM stubbed out so no network
calls are made and no API key is required.

Real PDFs are generated on-the-fly with reportlab so the tests stay
self-contained and fast.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from openpyxl import Workbook, load_workbook

from extractkit.checkpoint import load_or_create
from extractkit.config import Settings
from extractkit.excel_handler import count_data_rows
from extractkit.extractor import Extractor, ExtractorConfig
from extractkit.llm_client import LLMClient
from extractkit.schemas import EXCEL_COLUMNS, StructuredFields, SynthesisFields


def _write_minimal_pdf(path: Path, body_text: str) -> None:
    """Create a real PDF with extractable text content.

    The orchestrator rejects empty PDFs as scanned-only, so test PDFs
    must contain real text. ``reportlab`` is the standard tool for
    generating well-formed PDFs from Python.
    """
    from reportlab.pdfgen import canvas

    pdf = canvas.Canvas(str(path))
    pdf.drawString(72, 720, body_text)
    pdf.drawString(72, 700, "Sample academic article content for testing.")
    pdf.save()


def _make_template(path: Path) -> None:
    """Create an Excel template with the canonical headers."""
    workbook = Workbook()
    worksheet = workbook.active
    assert worksheet is not None
    worksheet.append(list(EXCEL_COLUMNS))
    workbook.save(path)


def _stub_llm(
    structured: StructuredFields,
    synthesis: SynthesisFields,
) -> LLMClient:
    """Build an ``LLMClient`` with both extraction methods stubbed.

    The returned client never touches OpenAI; calling either method
    returns the pre-built schema instance.
    """
    settings = Settings(OPENAI_API_KEY="sk-test-not-real")
    client = LLMClient(settings)
    client.extract_structured = MagicMock(  # type: ignore[method-assign]
        return_value=structured,
    )
    client.extract_synthesis = MagicMock(  # type: ignore[method-assign]
        return_value=synthesis,
    )
    return client


@pytest.fixture
def pdf_folder(tmp_path: Path) -> Path:
    """A folder with three small PDFs ready for processing."""
    folder = tmp_path / "papers"
    folder.mkdir()
    _write_minimal_pdf(folder / "paper_a.pdf", "Paper A content")
    _write_minimal_pdf(folder / "paper_b.pdf", "Paper B content")
    _write_minimal_pdf(folder / "paper_c.pdf", "Paper C content")
    return folder


@pytest.fixture
def template_path(tmp_path: Path) -> Path:
    """A freshly created Excel template at a stable path."""
    path = tmp_path / "template.xlsx"
    _make_template(path)
    return path


@pytest.fixture
def output_path(tmp_path: Path) -> Path:
    """Where the orchestrator should write its growing workbook."""
    return tmp_path / "output.xlsx"


@pytest.fixture
def checkpoint_path(tmp_path: Path) -> Path:
    """Where the orchestrator should write its checkpoint JSON."""
    return tmp_path / "output.xlsx.checkpoint.json"


def test_full_run_processes_every_pdf(
    pdf_folder: Path,
    template_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    structured_extraction_dict: dict[str, str],
    synthesis_extraction_dict: dict[str, str],
) -> None:
    """A clean run should process all PDFs and write one row each."""
    settings = Settings(OPENAI_API_KEY="sk-test-not-real")
    llm = _stub_llm(
        structured=StructuredFields(**structured_extraction_dict),
        synthesis=SynthesisFields(**synthesis_extraction_dict),
    )
    config = ExtractorConfig(
        pdf_folder=pdf_folder,
        template_path=template_path,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
    )

    summary = Extractor(settings=settings, config=config, llm_client=llm).run()

    assert summary.total == 3
    assert summary.succeeded == 3
    assert summary.skipped == 0
    assert summary.failed == {}
    assert count_data_rows(output_path) == 3


def test_rerun_skips_already_processed_pdfs(
    pdf_folder: Path,
    template_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    structured_extraction_dict: dict[str, str],
    synthesis_extraction_dict: dict[str, str],
) -> None:
    """A second run with the same checkpoint should skip everything done."""
    settings = Settings(OPENAI_API_KEY="sk-test-not-real")
    structured = StructuredFields(**structured_extraction_dict)
    synthesis = SynthesisFields(**synthesis_extraction_dict)
    config = ExtractorConfig(
        pdf_folder=pdf_folder,
        template_path=template_path,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
    )

    # First run: everything fresh.
    Extractor(
        settings=settings,
        config=config,
        llm_client=_stub_llm(structured, synthesis),
    ).run()

    # Second run with a freshly-built (still-stubbed) client. The
    # checkpoint should make the orchestrator skip every PDF.
    second_llm = _stub_llm(structured, synthesis)
    summary = Extractor(settings=settings, config=config, llm_client=second_llm).run()

    assert summary.total == 3
    assert summary.skipped == 3
    assert summary.succeeded == 0
    # The stubbed LLM must NOT have been invoked on the second run.
    second_llm.extract_structured.assert_not_called()  # type: ignore[attr-defined]
    second_llm.extract_synthesis.assert_not_called()  # type: ignore[attr-defined]


def test_failed_pdf_does_not_abort_the_run(
    pdf_folder: Path,
    template_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    structured_extraction_dict: dict[str, str],
    synthesis_extraction_dict: dict[str, str],
) -> None:
    """If one PDF's LLM call fails, the run continues for the others."""
    settings = Settings(OPENAI_API_KEY="sk-test-not-real")
    structured = StructuredFields(**structured_extraction_dict)
    synthesis = SynthesisFields(**synthesis_extraction_dict)

    llm = _stub_llm(structured, synthesis)

    # Make the second call to extract_structured raise; the other two
    # succeed. The orchestrator should record one failure and two wins.
    call_counter = {"n": 0}

    def flaky_structured(_text: str) -> StructuredFields:
        call_counter["n"] += 1
        if call_counter["n"] == 2:
            from extractkit.exceptions import LLMError

            raise LLMError("simulated transient failure")
        return structured

    llm.extract_structured = MagicMock(  # type: ignore[method-assign]
        side_effect=flaky_structured,
    )

    config = ExtractorConfig(
        pdf_folder=pdf_folder,
        template_path=template_path,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
    )
    summary = Extractor(settings=settings, config=config, llm_client=llm).run()

    assert summary.total == 3
    assert summary.succeeded == 2
    assert len(summary.failed) == 1
    assert count_data_rows(output_path) == 2


def test_template_with_wrong_headers_is_rejected(
    pdf_folder: Path,
    tmp_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    structured_extraction_dict: dict[str, str],
    synthesis_extraction_dict: dict[str, str],
) -> None:
    """A template missing schema columns must fail fast, not silently mis-write."""
    bad_template = tmp_path / "bad_template.xlsx"
    workbook = Workbook()
    worksheet = workbook.active
    assert worksheet is not None
    worksheet.append(["Wrong", "Headers", "Here"])
    workbook.save(bad_template)

    settings = Settings(OPENAI_API_KEY="sk-test-not-real")
    llm = _stub_llm(
        structured=StructuredFields(**structured_extraction_dict),
        synthesis=SynthesisFields(**synthesis_extraction_dict),
    )
    config = ExtractorConfig(
        pdf_folder=pdf_folder,
        template_path=bad_template,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
    )

    from extractkit.exceptions import ExcelError

    with pytest.raises(ExcelError):
        Extractor(settings=settings, config=config, llm_client=llm).run()


def test_output_rows_are_aligned_to_excel_columns(
    pdf_folder: Path,
    template_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    structured_extraction_dict: dict[str, str],
    synthesis_extraction_dict: dict[str, str],
) -> None:
    """Cells must end up under the matching header, not just in order."""
    settings = Settings(OPENAI_API_KEY="sk-test-not-real")
    llm = _stub_llm(
        structured=StructuredFields(**structured_extraction_dict),
        synthesis=SynthesisFields(**synthesis_extraction_dict),
    )
    config = ExtractorConfig(
        pdf_folder=pdf_folder,
        template_path=template_path,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
    )
    Extractor(settings=settings, config=config, llm_client=llm).run()

    workbook = load_workbook(output_path)
    worksheet = workbook.active
    assert worksheet is not None
    rows = list(worksheet.iter_rows(values_only=True))
    header_row = list(rows[0])
    data_row = list(rows[1])

    article_name_index = header_row.index("Article Name")
    year_index = header_row.index("Year")
    assert data_row[article_name_index] == structured_extraction_dict["article_name"]
    assert data_row[year_index] == structured_extraction_dict["year"]


def test_checkpoint_records_successes_and_failures(
    pdf_folder: Path,
    template_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    structured_extraction_dict: dict[str, str],
    synthesis_extraction_dict: dict[str, str],
) -> None:
    """The checkpoint file must reflect what actually happened."""
    settings = Settings(OPENAI_API_KEY="sk-test-not-real")
    structured = StructuredFields(**structured_extraction_dict)
    synthesis = SynthesisFields(**synthesis_extraction_dict)
    llm = _stub_llm(structured, synthesis)

    # Make extract_synthesis raise on the third call.
    call_counter = {"n": 0}

    def flaky_synthesis(_text: str) -> SynthesisFields:
        call_counter["n"] += 1
        if call_counter["n"] == 3:
            from extractkit.exceptions import LLMError

            raise LLMError("simulated transient failure")
        return synthesis

    llm.extract_synthesis = MagicMock(  # type: ignore[method-assign]
        side_effect=flaky_synthesis,
    )

    config = ExtractorConfig(
        pdf_folder=pdf_folder,
        template_path=template_path,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
    )
    Extractor(settings=settings, config=config, llm_client=llm).run()

    checkpoint = load_or_create(checkpoint_path)
    assert len(checkpoint.processed) == 2
    assert len(checkpoint.failed) == 1
