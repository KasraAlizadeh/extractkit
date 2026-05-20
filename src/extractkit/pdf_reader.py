"""PDF text extraction.

A thin wrapper around :mod:`pypdf` that turns a PDF file on disk into a
single string of text, plus a helper for listing the PDFs in a folder.

Keeping this isolated in one module means the rest of the package never
imports a PDF library directly, so swapping ``pypdf`` for ``pdfplumber``
or a cloud OCR service later is a one-file change.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError as PypdfReadError

from extractkit.exceptions import PDFReadError

# Page texts are joined with a form-feed character, which is the
# convention pdftotext uses. It is a single character the LLM treats as
# a clear page boundary without inflating the token count.
_PAGE_SEPARATOR = "\f"


def list_pdfs(folder: Path) -> list[Path]:
    """Return the PDFs sitting directly inside ``folder``.

    Only the top-level of the folder is scanned; subdirectories are
    ignored on purpose so the caller has full control over what gets
    processed.

    Args:
        folder: Directory to scan.

    Returns:
        A sorted list of PDF paths. Sorting makes runs deterministic,
        which matters for resumability and for tests.

    Raises:
        PDFReadError: If the folder does not exist or is not a directory.
    """
    if not folder.exists():
        raise PDFReadError(str(folder), "folder does not exist")
    if not folder.is_dir():
        raise PDFReadError(str(folder), "path is not a directory")

    # ``glob`` rather than ``rglob`` because we deliberately do not recurse.
    pdfs = [p for p in folder.glob("*.pdf") if p.is_file()]
    return sorted(pdfs)


def read_pdf_text(path: Path) -> str:
    """Extract all text from a PDF as a single string.

    Pages are concatenated in order, separated by a form-feed character.
    Pages that fail to extract individually are skipped rather than
    aborting the whole document: a corrupted page near the end of a
    50-page paper should not throw away the other 49.

    Args:
        path: Path to the PDF file.

    Returns:
        The full document text. May be empty for image-only (scanned) PDFs.

    Raises:
        PDFReadError: If the file cannot be opened or parsed at all.
    """
    if not path.exists():
        raise PDFReadError(str(path), "file does not exist")
    if not path.is_file():
        raise PDFReadError(str(path), "path is not a file")

    try:
        reader = PdfReader(str(path))
    except (PypdfReadError, OSError, ValueError) as exc:
        raise PDFReadError(str(path), f"could not open PDF: {exc}") from exc

    page_texts: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            # Skip the bad page; the rest of the document is still useful.
            text = ""
        page_texts.append(text)

    return _PAGE_SEPARATOR.join(page_texts)


def read_pdf_with_metadata(path: Path) -> tuple[str, int]:
    """Like :func:`read_pdf_text` but also returns the page count.

    Useful for logging and for cost estimation before the LLM call.

    Args:
        path: Path to the PDF file.

    Returns:
        A ``(text, page_count)`` tuple.

    Raises:
        PDFReadError: If the file cannot be opened or parsed.
    """
    text = read_pdf_text(path)
    # ``text`` contains one separator per page boundary, so the page count
    # is one more than the number of separators (when there is text at all).
    page_count = text.count(_PAGE_SEPARATOR) + 1 if text else 0
    return text, page_count
