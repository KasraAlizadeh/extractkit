"""Excel template reader and incremental row writer.

The Excel workbook plays two roles:

* On input it is a *template* — its column headers define the extraction
  schema (which fields the LLM is asked to find).
* On output the same workbook is written back with one new row per PDF
  appended below the headers.

Writes are *incremental*: after every PDF, the workbook is saved to a
temporary file and then atomically renamed over the previous version.
That way a crash, a kill -9, or even a power loss never leaves you with
a half-written file or with hours of LLM calls thrown away.
"""

from __future__ import annotations

import os
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException

from extractkit.exceptions import ExcelError


def read_column_headers(template_path: Path, sheet: str | None = None) -> list[str]:
    """Read the column headers from the first row of the template.

    Args:
        template_path: Path to the ``.xlsx`` template.
        sheet: Sheet name to read. Defaults to the active sheet.

    Returns:
        A list of header strings in column order. Trailing empty cells
        are dropped so the list length matches the real schema width.

    Raises:
        ExcelError: If the file cannot be opened or the header row is empty.
    """
    if not template_path.exists():
        raise ExcelError(f"Excel template not found: {template_path}")

    try:
        workbook = load_workbook(template_path, read_only=True, data_only=True)
    except (InvalidFileException, OSError) as exc:
        raise ExcelError(f"Could not open Excel file '{template_path}': {exc}") from exc

    try:
        worksheet = workbook[sheet] if sheet else workbook.active
        if worksheet is None:
            raise ExcelError("Workbook has no active sheet")

        first_row = next(worksheet.iter_rows(min_row=1, max_row=1, values_only=True), None)
        if first_row is None:
            raise ExcelError("Excel template has no header row")

        # Strip trailing empty cells: openpyxl pads rows with ``None`` to
        # the worksheet's declared width, which would otherwise inflate
        # the schema with phantom columns.
        headers = [
            str(cell).strip() for cell in first_row if cell is not None and str(cell).strip()
        ]
        if not headers:
            raise ExcelError("Excel template header row is empty")

        return headers
    finally:
        workbook.close()


def append_row(
    template_path: Path,
    output_path: Path,
    row: list[str],
    sheet: str | None = None,
) -> None:
    """Append a single row to the workbook and save atomically.

    On the first call the template is copied to ``output_path``. On
    subsequent calls ``output_path`` itself is opened and appended to,
    so the file grows one row at a time.

    Args:
        template_path: Source template (used only when the output does
            not yet exist).
        output_path: Destination workbook to grow.
        row: Cell values for the new row, in column order.
        sheet: Sheet to append to. Defaults to the active sheet.

    Raises:
        ExcelError: If the workbook cannot be opened, modified, or saved.
    """
    # The source is the in-progress output if it exists, otherwise the
    # pristine template. This is what makes the writer incremental.
    source_path = output_path if output_path.exists() else template_path

    if not source_path.exists():
        raise ExcelError(f"Neither output nor template exists: {source_path}")

    try:
        workbook = load_workbook(source_path)
    except (InvalidFileException, OSError) as exc:
        raise ExcelError(f"Could not open workbook '{source_path}': {exc}") from exc

    try:
        worksheet = workbook[sheet] if sheet else workbook.active
        if worksheet is None:
            raise ExcelError("Workbook has no active sheet")

        worksheet.append(row)

        # Save to a sibling temp file, then atomically replace the target.
        # ``os.replace`` is atomic on POSIX and on Windows, so the output
        # path is either the old workbook or the new one — never a
        # half-written file.
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        workbook.save(tmp_path)
        os.replace(tmp_path, output_path)
    except OSError as exc:
        raise ExcelError(f"Could not save workbook '{output_path}': {exc}") from exc
    finally:
        workbook.close()


def count_data_rows(workbook_path: Path, sheet: str | None = None) -> int:
    """Return the number of populated data rows (excluding the header).

    Used at startup to report progress when resuming a previous run.

    Args:
        workbook_path: Path to an existing workbook.
        sheet: Sheet name. Defaults to the active sheet.

    Returns:
        Zero if the file does not yet exist, otherwise the data-row count.

    Raises:
        ExcelError: If the file exists but cannot be opened.
    """
    if not workbook_path.exists():
        return 0

    try:
        workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    except (InvalidFileException, OSError) as exc:
        raise ExcelError(f"Could not open workbook '{workbook_path}': {exc}") from exc

    try:
        worksheet = workbook[sheet] if sheet else workbook.active
        if worksheet is None:
            return 0
        # ``max_row`` includes the header row, so subtract one.
        # The ``int(...)`` cast is for mypy: openpyxl's stubs type
        # ``max_row`` as ``Any``, but it is always an integer at runtime.
        return max(int(worksheet.max_row) - 1, 0)
    finally:
        workbook.close()
