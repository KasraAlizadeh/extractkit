"""Top-level extraction pipeline.

:class:`Extractor` is the only thing the CLI needs to call. It owns the
loop over PDFs and decides, for each one, whether to skip it (already
processed), how to fan out the two LLM passes, how to write the result
into the Excel workbook, and how to record success or failure in the
checkpoint.

The orchestrator is deliberately the only place that knows about *all*
the lower-level modules; every other module knows only its own concern.
Keeping that boundary clean means each module can be tested in
isolation and swapped out independently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from extractkit.checkpoint import load_or_create
from extractkit.config import Settings
from extractkit.excel_handler import append_row, read_column_headers
from extractkit.exceptions import (
    ExcelError,
    ExtractionError,
    LLMError,
    PDFReadError,
)
from extractkit.llm_client import LLMClient
from extractkit.pdf_reader import list_pdfs, read_pdf_text
from extractkit.schemas import (
    EXCEL_COLUMNS,
    ArticleExtraction,
    extraction_to_row,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractorConfig:
    """Per-run configuration for the orchestrator.

    Attributes:
        pdf_folder: Directory of PDFs to process (top level only).
        template_path: Excel template whose header row defines the schema.
        output_path: Destination workbook (grown incrementally).
        checkpoint_path: JSON checkpoint file (created if absent).
        sheet: Worksheet name, or ``None`` for the active sheet.
    """

    pdf_folder: Path
    template_path: Path
    output_path: Path
    checkpoint_path: Path
    sheet: str | None = None


@dataclass
class RunSummary:
    """Outcome of a full :meth:`Extractor.run` call.

    Attributes:
        total: Number of PDFs found in the folder.
        skipped: PDFs skipped because the checkpoint already had them.
        succeeded: PDFs extracted successfully in this run.
        failed: PDFs that failed in this run, with their reason.
        started_at: Run start time (ISO-8601, UTC).
        finished_at: Run end time (ISO-8601, UTC).
    """

    total: int
    skipped: int
    succeeded: int
    failed: dict[str, str]
    started_at: str
    finished_at: str


class Extractor:
    """Drive the end-to-end extraction pipeline.

    A single :class:`Extractor` instance handles one run. Construct it
    with everything it needs, then call :meth:`run`.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        config: ExtractorConfig,
        llm_client: LLMClient | None = None,
        console: Console | None = None,
    ) -> None:
        """Build an extractor.

        ``llm_client`` and ``console`` are injectable so tests can
        substitute a fake LLM and a silent console.
        """
        self._settings = settings
        self._config = config
        self._llm = llm_client or LLMClient(settings)
        self._console = console or Console()

    def run(self) -> RunSummary:
        """Execute the full pipeline and return a summary.

        Validates inputs, loads or creates the checkpoint, iterates the
        PDFs, and writes the output workbook one row at a time.

        Returns:
            A :class:`RunSummary` describing what happened.
        """
        self._settings.validate_ready()
        self._validate_schema_matches_template()

        pdfs = list_pdfs(self._config.pdf_folder)
        checkpoint = load_or_create(self._config.checkpoint_path)
        started_at = datetime.now(UTC).isoformat()

        skipped = 0
        succeeded = 0

        with self._progress() as progress:
            task = progress.add_task("Extracting", total=len(pdfs))
            for pdf_path in pdfs:
                progress.update(task, description=f"Extracting {pdf_path.name}")
                if checkpoint.is_done(pdf_path.name):
                    skipped += 1
                    progress.advance(task)
                    continue

                try:
                    self._process_one(pdf_path)
                except ExtractionError as exc:
                    checkpoint.mark_failed(pdf_path.name, exc.reason)
                    self._console.print(f"[red]✗[/red] {pdf_path.name}: {exc.reason}")
                else:
                    checkpoint.mark_done(pdf_path.name)
                    succeeded += 1

                progress.advance(task)

        return RunSummary(
            total=len(pdfs),
            skipped=skipped,
            succeeded=succeeded,
            failed=dict(checkpoint.failed),
            started_at=started_at,
            finished_at=datetime.now(UTC).isoformat(),
        )

    def _process_one(self, pdf_path: Path) -> None:
        """Run both LLM passes for one PDF and append the row.

        Raises:
            ExtractionError: For any per-document failure. Wrapping the
                original exception keeps the orchestrator's catch-block
                simple (one type to catch) without losing the cause.
        """
        try:
            text = read_pdf_text(pdf_path)
        except PDFReadError as exc:
            raise ExtractionError(pdf_path.name, exc.reason) from exc

        if not text.strip():
            raise ExtractionError(
                pdf_path.name,
                "no extractable text (likely a scanned PDF; OCR is out of scope)",
            )

        try:
            structured = self._llm.extract_structured(text)
            synthesis = self._llm.extract_synthesis(text)
        except LLMError as exc:
            raise ExtractionError(pdf_path.name, str(exc)) from exc

        extraction = ArticleExtraction(structured=structured, synthesis=synthesis)
        row = extraction_to_row(extraction)

        try:
            append_row(
                template_path=self._config.template_path,
                output_path=self._config.output_path,
                row=row,
                sheet=self._config.sheet,
            )
        except ExcelError as exc:
            # The LLM work is already done, so losing the row to a write
            # failure is the worst possible outcome: surface loudly.
            raise ExtractionError(pdf_path.name, f"failed to write row: {exc}") from exc

    def _validate_schema_matches_template(self) -> None:
        """Check that the template's headers match the package's schema.

        Catches drift between the Excel template the user supplies and
        the 33 columns the Pydantic schemas were built for. Without this
        check, a renamed header would silently send the wrong values
        into the wrong column.

        Raises:
            ExcelError: If the headers do not match.
        """
        headers = read_column_headers(self._config.template_path, self._config.sheet)
        if tuple(headers) != EXCEL_COLUMNS:
            missing = [c for c in EXCEL_COLUMNS if c not in headers]
            extra = [h for h in headers if h not in EXCEL_COLUMNS]
            raise ExcelError(
                "Excel template headers do not match the extraction schema. "
                f"Missing: {missing or 'none'}. Unexpected: {extra or 'none'}."
            )

    def _progress(self) -> Progress:
        """Build the rich progress bar used during the run."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self._console,
            transient=False,
        )
