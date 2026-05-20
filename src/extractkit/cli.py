"""Command-line interface.

Exposes the :func:`extract` command via Typer so users can run::

    extractkit extract \\
        --pdfs ./papers \\
        --template ./template.xlsx \\
        --output ./results.xlsx

The CLI is intentionally thin: it parses arguments, builds an
:class:`extractkit.extractor.ExtractorConfig`, hands off to
:class:`extractkit.extractor.Extractor`, and prints a final summary.
All real work lives in the package modules so the same pipeline can be
driven from a notebook, a Streamlit app, or a test — without ever
touching this file.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from extractkit.config import load_settings
from extractkit.exceptions import ExtractKitError
from extractkit.extractor import Extractor, ExtractorConfig, RunSummary

app = typer.Typer(
    name="extractkit",
    help="LLM-powered structured extraction from academic PDFs into Excel.",
    no_args_is_help=True,
    add_completion=False,
)


_console = Console()


def _setup_logging(verbose: bool) -> None:
    """Route logging through Rich so log lines coexist with the progress bar.

    Debug-level logs are noisy (HTTPX request tracing in particular), so
    we keep them at INFO unless ``--verbose`` is given.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=_console, rich_tracebacks=True, show_path=False)],
    )
    # The OpenAI and httpx libraries are noisy at DEBUG; cap them at INFO
    # so ``--verbose`` shows our own debug output without drowning in
    # third-party noise.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)


def _render_summary(summary: RunSummary, output_path: Path) -> None:
    """Print a human-friendly final report after a run."""
    table = Table(title="Extraction Summary", title_style="bold cyan", expand=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Total PDFs", str(summary.total))
    table.add_row("Skipped (already done)", str(summary.skipped))
    table.add_row("Succeeded this run", str(summary.succeeded))
    table.add_row("Failed", str(len(summary.failed)))
    table.add_row("Output", str(output_path))
    _console.print(table)

    if summary.failed:
        failures = Table(title="Failures", title_style="bold red", expand=False)
        failures.add_column("PDF", style="bold")
        failures.add_column("Reason")
        for name, reason in summary.failed.items():
            failures.add_row(name, reason)
        _console.print(failures)


@app.command()
def extract(
    pdfs: Annotated[
        Path,
        typer.Option(
            "--pdfs",
            "-p",
            help="Folder containing the PDF articles to process.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    template: Annotated[
        Path,
        typer.Option(
            "--template",
            "-t",
            help="Excel template whose header row defines the schema.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Where to write the filled workbook (created or grown).",
            resolve_path=True,
        ),
    ],
    checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint",
            "-c",
            help="Checkpoint JSON path. Defaults to '<output>.checkpoint.json'.",
            resolve_path=True,
        ),
    ] = None,
    sheet: Annotated[
        str | None,
        typer.Option(
            "--sheet",
            "-s",
            help="Worksheet name to use. Defaults to the active sheet.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable debug-level logging.",
        ),
    ] = False,
) -> None:
    """Extract structured data from a folder of PDFs into an Excel workbook."""
    _setup_logging(verbose)

    # Derive the checkpoint path from the output path when the user did
    # not specify one. Keeping them side-by-side means a re-run with the
    # same ``--output`` automatically resumes.
    checkpoint_path = checkpoint or output.with_suffix(output.suffix + ".checkpoint.json")

    config = ExtractorConfig(
        pdf_folder=pdfs,
        template_path=template,
        output_path=output,
        checkpoint_path=checkpoint_path,
        sheet=sheet,
    )

    _console.print(
        Panel.fit(
            f"[bold]PDFs:[/bold]       {pdfs}\n"
            f"[bold]Template:[/bold]   {template}\n"
            f"[bold]Output:[/bold]     {output}\n"
            f"[bold]Checkpoint:[/bold] {checkpoint_path}",
            title="extractkit",
            border_style="cyan",
        )
    )

    try:
        settings = load_settings()
        extractor = Extractor(settings=settings, config=config, console=_console)
        summary = extractor.run()
    except ExtractKitError as exc:
        _console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    _render_summary(summary, output)

    # Non-zero exit when any PDF failed, so CI scripts and shell loops
    # notice. The Excel output is still valid for the ones that worked.
    if summary.failed:
        raise typer.Exit(code=2)


def main() -> None:
    """Entry point for the installed ``extractkit`` console script."""
    try:
        app()
    except KeyboardInterrupt:
        _console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(130)  # 128 + SIGINT, the conventional Ctrl-C exit code


if __name__ == "__main__":
    main()
