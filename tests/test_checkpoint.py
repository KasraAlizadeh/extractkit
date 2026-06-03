"""Tests for the JSON checkpoint module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from extractkit.checkpoint import Checkpoint, load_or_create
from extractkit.exceptions import ExtractKitError


def test_load_or_create_returns_empty_when_file_absent(tmp_path: Path) -> None:
    path = tmp_path / "run.checkpoint.json"
    checkpoint = load_or_create(path)
    assert checkpoint.processed == set()
    assert checkpoint.failed == {}


def test_mark_done_persists_to_disk(tmp_path: Path) -> None:
    path = tmp_path / "run.checkpoint.json"
    checkpoint = Checkpoint(path=path)
    checkpoint.mark_done("paper1.pdf")
    checkpoint.mark_done("paper2.pdf")

    reloaded = load_or_create(path)
    assert reloaded.is_done("paper1.pdf")
    assert reloaded.is_done("paper2.pdf")
    assert not reloaded.is_done("paper3.pdf")


def test_mark_failed_then_done_clears_failure(tmp_path: Path) -> None:
    """A retried PDF that finally works should leave the failures list."""
    path = tmp_path / "run.checkpoint.json"
    checkpoint = Checkpoint(path=path)
    checkpoint.mark_failed("paper1.pdf", "transient API error")
    assert "paper1.pdf" in checkpoint.failed

    checkpoint.mark_done("paper1.pdf")
    assert "paper1.pdf" not in checkpoint.failed
    assert checkpoint.is_done("paper1.pdf")


def test_corrupt_checkpoint_raises(tmp_path: Path) -> None:
    """Refuse to silently start over when the file is unreadable."""
    path = tmp_path / "run.checkpoint.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ExtractKitError):
        load_or_create(path)


def test_wrong_version_raises(tmp_path: Path) -> None:
    """Future schema changes must not silently misread old checkpoints."""
    path = tmp_path / "run.checkpoint.json"
    path.write_text(
        json.dumps({"version": 999, "processed": [], "failed": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ExtractKitError):
        load_or_create(path)
