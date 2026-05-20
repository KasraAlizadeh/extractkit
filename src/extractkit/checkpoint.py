"""Run checkpointing for crash-resilient batch processing.

A run can take hours and hundreds of API calls; we cannot afford to
restart from scratch after every transient failure. The checkpoint
records which PDFs have already been processed so a re-run picks up
from where the previous attempt left off.

Design choices:

* The checkpoint is a small JSON file living next to the output Excel
  workbook. JSON keeps it human-readable and easy to inspect or edit by
  hand if something goes wrong.
* Writes are atomic (temp file plus :func:`os.replace`) for the same
  reason as the Excel writer: a crash mid-write must not corrupt the
  checkpoint.
* The processed set is keyed by **filename**, not by absolute path, so
  moving the PDF folder does not invalidate the checkpoint.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from extractkit.exceptions import ExtractKitError

# Version stamp on every checkpoint so a future schema change (e.g.
# tracking per-file token counts) can be detected and migrated cleanly
# instead of silently misreading old files.
_CHECKPOINT_VERSION: int = 1


@dataclass
class Checkpoint:
    """In-memory view of a run's checkpoint.

    Attributes:
        path: Where this checkpoint is persisted.
        processed: Filenames already extracted successfully.
        failed: Mapping of filename to the last error message.
        started_at: ISO-8601 timestamp of the original run start.
    """

    path: Path
    processed: set[str] = field(default_factory=set)
    failed: dict[str, str] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def mark_done(self, filename: str) -> None:
        """Record a PDF as successfully processed and persist.

        Successful processing clears any previous failure for the same
        file: a retried PDF that finally worked should not stay in the
        failures list.
        """
        self.processed.add(filename)
        self.failed.pop(filename, None)
        self.save()

    def mark_failed(self, filename: str, reason: str) -> None:
        """Record a PDF as failed (with a short reason) and persist."""
        self.failed[filename] = reason
        self.save()

    def is_done(self, filename: str) -> bool:
        """Return ``True`` if this filename was already processed successfully."""
        return filename in self.processed

    def save(self) -> None:
        """Persist the checkpoint to disk atomically.

        Raises:
            ExtractKitError: If the file cannot be written.
        """
        payload: dict[str, Any] = {
            "version": _CHECKPOINT_VERSION,
            "started_at": self.started_at,
            "updated_at": datetime.now(UTC).isoformat(),
            "processed": sorted(self.processed),
            "failed": self.failed,
        }
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            tmp_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(tmp_path, self.path)
        except OSError as exc:
            raise ExtractKitError(f"Could not write checkpoint '{self.path}': {exc}") from exc


def load_or_create(path: Path) -> Checkpoint:
    """Load an existing checkpoint or return a fresh one.

    If the file exists but cannot be parsed, the user is informed via a
    raised exception rather than silently starting over — silently
    discarding hours of work would be a much worse default.

    Args:
        path: Where the checkpoint lives (or will live).

    Returns:
        A :class:`Checkpoint` ready to use.

    Raises:
        ExtractKitError: If an existing checkpoint cannot be read.
    """
    if not path.exists():
        return Checkpoint(path=path)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExtractKitError(
            f"Could not read checkpoint '{path}': {exc}. "
            "Delete the file manually if you want to start a fresh run."
        ) from exc

    version = raw.get("version")
    if version != _CHECKPOINT_VERSION:
        raise ExtractKitError(
            f"Checkpoint '{path}' has version {version!r}, expected "
            f"{_CHECKPOINT_VERSION}. Delete it to start a fresh run."
        )

    return Checkpoint(
        path=path,
        processed=set(raw.get("processed", [])),
        failed=dict(raw.get("failed", {})),
        started_at=str(raw.get("started_at", datetime.now(UTC).isoformat())),
    )
