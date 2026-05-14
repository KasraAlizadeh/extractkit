"""Custom exceptions for extractkit.

Defining specific exception types (instead of raising bare ``Exception``)
lets callers catch exactly the failure they care about and lets us attach
useful context to each error.
"""

from __future__ import annotations


class ExtractKitError(Exception):
    """Base class for all extractkit errors.

    Catching this catches every error the package raises deliberately.
    """


class ConfigError(ExtractKitError):
    """Raised when configuration is missing or invalid (e.g. no API key)."""


class PDFReadError(ExtractKitError):
    """Raised when a PDF cannot be opened or its text cannot be extracted."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to read PDF '{path}': {reason}")


class ExcelError(ExtractKitError):
    """Raised when the Excel template cannot be read or written."""


class LLMError(ExtractKitError):
    """Raised when the LLM API call fails or returns an unusable response."""


class ExtractionError(ExtractKitError):
    """Raised when extraction for a single document fails.

    The orchestrator catches this per-document so one bad PDF does not
    abort the whole run.
    """

    def __init__(self, source: str, reason: str) -> None:
        self.source = source
        self.reason = reason
        super().__init__(f"Extraction failed for '{source}': {reason}")
