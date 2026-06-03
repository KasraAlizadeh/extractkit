"""Tests for the PDF reader module."""

from __future__ import annotations

from pathlib import Path

import pytest

from extractkit.exceptions import PDFReadError
from extractkit.pdf_reader import list_pdfs, read_pdf_text


def test_list_pdfs_empty_folder(tmp_path: Path) -> None:
    assert list_pdfs(tmp_path) == []


def test_list_pdfs_returns_sorted_pdfs_only(tmp_path: Path) -> None:
    """Only ``.pdf`` files, returned in deterministic order."""
    (tmp_path / "b.pdf").touch()
    (tmp_path / "a.pdf").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "report.PDF").touch()  # uppercase extension is ignored

    result = list_pdfs(tmp_path)
    names = [p.name for p in result]
    assert names == ["a.pdf", "b.pdf"]


def test_list_pdfs_ignores_subfolders(tmp_path: Path) -> None:
    """Only the top level is scanned, by design."""
    (tmp_path / "top.pdf").touch()
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.pdf").touch()

    names = [p.name for p in list_pdfs(tmp_path)]
    assert names == ["top.pdf"]


def test_list_pdfs_missing_folder_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(PDFReadError):
        list_pdfs(missing)


def test_read_pdf_text_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(PDFReadError):
        read_pdf_text(tmp_path / "missing.pdf")


def test_read_pdf_text_bad_file_raises(tmp_path: Path) -> None:
    """A file that exists but is not a real PDF should raise PDFReadError."""
    bad = tmp_path / "bad.pdf"
    bad.write_text("this is not a PDF", encoding="utf-8")
    with pytest.raises(PDFReadError):
        read_pdf_text(bad)
