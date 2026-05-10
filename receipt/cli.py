"""Typer CLI — entry point: `receipt`."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.theme import Theme

app = typer.Typer(
    name="receipt",
    help="Intelligent personal finance analysis engine.",
    add_completion=False,
    rich_markup_mode="rich",
)

_theme = Theme(
    {
        "income": "bold green",
        "expense": "bold red",
        "warning": "bold yellow",
        "info": "dim cyan",
        "critical": "bold red on white",
        "header": "bold white on blue",
    }
)
console = Console(theme=_theme)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _get_api_key(api_key: str | None) -> str:
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        console.print("[warning]No Anthropic API key — narrative generation will be skipped.[/warning]")
    return key


def _run_pipeline(
    file_path: Path,
    fmt: str,
    period: int,
    compare: bool,
    save: bool,
    api_key: str | None,
    output: str,
) -> None:
    _load_env()
    resolved_key = _get_api_key(api_key)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        # --- Ingest ---
        task = progress.add_task("Detecting parser…", total=None)
        from receipt.ingestion import detect_parser
        from receipt.ingestion.chase import ChaseParser
        from receipt.ingestion.bofa import BofAParser
        from receipt.ingestion.plaid import PlaidParser
        from receipt.ingestion.csv_parser import GenericCSVParser

        parser_map = {
            "chase": ChaseParser,
            "bofa": BofAParser,
            "plaid": PlaidParser,
            "generic": GenericCSVParser,
        }
        if fmt == "auto":
            parser = detect_parser(file_path)
        else:
            parser = parser_map[fmt]()

        progress.update(task, description=f"Parsing with {type(parser).__name__}…")
        df = parser.parse(file_path)

        # Filter to period
        cutoff = datetime.now(timezone.utc) - timedelta(days=period)
        df = df[df["date"] >= cutoff].copy()

        if df.empty:
            console.print(f"[warning]No transactions found in the last {period} days.[/warning]")
            raise typer.Exit(1)

        # --- Clean ---
        progress.update(task, description="Cleaning data…")
        from receipt.pipeline.cleaner import normalize_descriptions, deduplicate, normalize_dates

        df = normalize_descriptions(df)
        df = deduplicate(df)
        df = normalize_dates(df)

        # --- Categorize ---
        progress.update(task, description="Categorising transactions…")
        from receipt.pipeline.categorizer import SemanticCategorizer

        categorizer = SemanticCategorizer()
        df = categorizer.categorize(df)

        # --- Anomalies ---
        progress.update(task, description="Detecting anomalies…")
        from receipt.analysis.anomalies import AnomalyDetector

        detector = AnomalyDetector()
        df = detector.fit_predict(df)

        # --- Stats ---
        progress.update(task, description="Computing statistics…")
        from receipt.pipeline.aggregator import compute_stats

        stats = compute_stats(df)

        # --- Patterns ---
        progress.update(task, description="Detecting patterns…")
        from receipt.analysis.patterns import detect_patterns

        patterns = detect_patterns(df)

        # --- Drift ---
        drift = None
        if compare:
            progress.update(task, description="Computing drift…")
            from receipt.storage.store import ReceiptStore
            from receipt.pipeline.drift import DriftDetector

            store = ReceiptStore()
            prev_df = store.get_previous_period(df["date"].min())
            if prev_df is not None and not prev_df.empty:
                detector_drift = DriftDetector()
                drift = detector_drift.compare_periods(df, prev_df)
            else:
                console.print("[info]No previous period found for comparison.[/info]")

        # --- Narrative ---
        narrative = None
        if resolved_key:
            progress.update(task, description="Generating narrative…")
            from receipt.analysis.narrator import Narrator

            narrator = Narrator(api_key=resolved_key)
            try:
                narrative = narrator.generate_narrative(stats, patterns, drift)
            except Exception as exc:
                console.print(f"[warning]Narrative generation failed: {exc}[/warning]")

        # --- Save ---
        run_id = None
        if save:
            progress.update(task, description="Saving to database…")
            from receipt.storage.store import ReceiptStore

            store = ReceiptStore()
            run_id = store.save_analysis(
                period_start=df["date"].min().to_pydatetime(),
                period_end=df["date"].max().to_pydatetime(),
                transaction_count=len(df),
                source_file=str(file_path),
                narrative=narrative.to_dict() if narrative else None,
            )
            store.save_transactions(df, run_id)
            store.upsert_merchants(df)

        progress.update(task, description="Done!")

    # --- Render output ---
    if output == "json":
        result = {
            "run_id": run_id,
            "stats": stats,
            "patterns": [
                {"type": p.type, "headline": p.headline, "severity": p.severity, "data": p.data}
                for p in patterns
            ],
            "drift": drift.to_dict() if drift else None,
            "narrative": narrative.to_dict() if narrative else None,
        }
        typer.echo(json.dumps(result, indent=2, default=str))
        return

    if output == "markdown":
        _render_markdown(stats, patterns, drift, narrative, run_id)
        return

    # Default: rich terminal
    _render_terminal(stats, patterns, drift, narrative, run_id, len(df))


def _render_terminal(stats, patterns, drift, narrative, run_id, tx_count) -> None:
    console.rule("[header] receipt analysis [/header]")

    # Summary table
    table = Table(title="Spending Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Transactions", str(tx_count))
    table.add_row("Total Spent", f"[expense]${abs(stats['total_spent']):.2f}[/expense]")
    table.add_row("Total Income", f"[income]${stats['total_income']:.2f}[/income]")
    table.add_row("Net", f"[{'income' if stats['net'] >= 0 else 'expense'}]${stats['net']:.2f}[/]")
    table.add_row("Subscriptions", f"${stats['subscription_total']:.2f}")
    if stats.get("most_frequent_merchant"):
        mf = stats["most_frequent_merchant"]
        table.add_row("Most Visited", f"{mf['merchant']} ({mf['count']}×)")
    if stats.get("largest_single_transaction"):
        lt = stats["largest_single_transaction"]
        table.add_row("Largest Purchase", f"${abs(lt['amount']):.2f} at {lt['description']}")
    console.print(table)

    # Category breakdown
    cat_table = Table(title="By Category", show_header=True, header_style="bold cyan")
    cat_table.add_column("Category")
    cat_table.add_column("Total", justify="right")
    cat_table.add_column("Txns", justify="right")
    cat_table.add_column("Avg", justify="right")
    for cat, data in sorted(stats["by_category"].items(), key=lambda x: -x[1]["total"]):
        cat_table.add_row(
            cat.replace("_", " ").title(),
            f"${data['total']:.2f}",
            str(data["count"]),
            f"${data['avg']:.2f}",
        )
    console.print(cat_table)

    # Patterns
    if patterns:
        console.print()
        console.print("[bold]Detected Patterns[/bold]")
        for p in patterns:
            color = {"info": "cyan", "warning": "yellow", "critical": "red"}.get(p.severity, "white")
            console.print(
                Panel(p.headline, title=f"[{color}]{p.type}[/{color}]", border_style=color)
            )

    # Drift
    if drift and drift.narrative_hints:
        console.print()
        console.print(Panel("\n".join(drift.narrative_hints), title="[bold]Drift vs Previous Period[/bold]", border_style="blue"))

    # Narrative
    if narrative:
        console.print()
        console.rule("[bold green] AI Insights [/bold green]")
        console.print(Panel(f"[bold]{narrative.tldr}[/bold]", border_style="green", title="TL;DR"))
        for insight in narrative.insights:
            console.print(
                Panel(
                    f"[bold]{insight.headline}[/bold]\n\n{insight.detail}",
                    border_style="bright_blue",
                )
            )
        console.print(Panel(narrative.next_steps, title="Next Steps", border_style="dim green"))

    if run_id:
        console.print(f"\n[info]Saved as run [bold]{run_id}[/bold]. Run `receipt history` to view past analyses.[/info]")


def _render_markdown(stats, patterns, drift, narrative, run_id) -> None:
    lines = ["# Receipt Analysis Report", ""]
    lines.append("## Summary")
    lines.append(f"- **Total Spent:** ${abs(stats['total_spent']):.2f}")
    lines.append(f"- **Total Income:** ${stats['total_income']:.2f}")
    lines.append(f"- **Net:** ${stats['net']:.2f}")
    lines.append(f"- **Subscriptions:** ${stats['subscription_total']:.2f}")
    lines.append("")
    lines.append("## By Category")
    for cat, data in sorted(stats["by_category"].items(), key=lambda x: -x[1]["total"]):
        lines.append(f"- **{cat}:** ${data['total']:.2f} ({data['count']} transactions)")
    if patterns:
        lines.append("")
        lines.append("## Patterns")
        for p in patterns:
            lines.append(f"- **[{p.severity.upper()}] {p.type}:** {p.headline}")
    if drift and drift.narrative_hints:
        lines.append("")
        lines.append("## Drift vs Previous Period")
        for hint in drift.narrative_hints:
            lines.append(f"- {hint}")
    if narrative:
        lines.append("")
        lines.append("## AI Insights")
        lines.append(f"> {narrative.tldr}")
        lines.append("")
        for insight in narrative.insights:
            lines.append(f"### {insight.headline}")
            lines.append(insight.detail)
            lines.append("")
        lines.append("## Next Steps")
        lines.append(narrative.next_steps)
    typer.echo("\n".join(lines))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def analyze(
    file: Annotated[Path, typer.Argument(help="CSV transaction file to analyze.")] = Path(""),
    fmt: Annotated[
        str, typer.Option("--format", "-f", help="Parser: auto|chase|bofa|plaid|generic")
    ] = "auto",
    period: Annotated[int, typer.Option("--period", "-p", help="Days to analyze (30/60/90)")] = 30,
    compare: Annotated[bool, typer.Option("--compare", help="Compare to previous period")] = False,
    output: Annotated[
        str, typer.Option("--output", "-o", help="Output format: terminal|json|markdown")
    ] = "terminal",
    save: Annotated[bool, typer.Option("--save", help="Save results to local DB")] = False,
    api_key: Annotated[Optional[str], typer.Option("--api-key", help="Anthropic API key")] = None,
) -> None:
    """Analyze a bank transaction CSV and generate insights."""
    _load_env()

    if not file or not file.exists():
        # Try sample data
        sample = Path(__file__).parent.parent / "data" / "sample" / "chase_sample.csv"
        if sample.exists() and (not file or str(file) == ""):
            console.print(f"[info]No file specified — using sample data: {sample}[/info]")
            file = sample
        else:
            console.print(f"[expense]File not found: {file}[/expense]")
            raise typer.Exit(1)

    if fmt not in ("auto", "chase", "bofa", "plaid", "generic"):
        console.print(f"[expense]Unknown format: {fmt}. Use auto|chase|bofa|plaid|generic[/expense]")
        raise typer.Exit(1)

    _run_pipeline(file, fmt, period, compare, save, api_key, output)


@app.command()
def history() -> None:
    """List past analysis runs."""
    _load_env()
    from receipt.storage.store import ReceiptStore

    store = ReceiptStore()
    runs = store.get_analysis_history()

    if not runs:
        console.print("[info]No analysis runs found. Run `receipt analyze` first.[/info]")
        return

    table = Table(title="Analysis History", show_header=True, header_style="bold")
    table.add_column("Run ID", style="dim")
    table.add_column("Date")
    table.add_column("Period")
    table.add_column("Txns", justify="right")
    table.add_column("TL;DR")

    for r in runs:
        period_str = f"{r['period_start'][:10]} – {r['period_end'][:10]}"
        table.add_row(
            r["run_id"],
            r["created_at"][:10],
            period_str,
            str(r["transaction_count"]),
            (r["tldr"] or "")[:60],
        )
    console.print(table)


@app.command()
def merchants() -> None:
    """Show top merchants by spend across all stored data."""
    _load_env()
    from receipt.storage.store import ReceiptStore

    store = ReceiptStore()
    data = store.get_merchants(limit=30)

    if not data:
        console.print("[info]No merchant data found. Run `receipt analyze --save` first.[/info]")
        return

    table = Table(title="Top Merchants", show_header=True, header_style="bold magenta")
    table.add_column("Merchant")
    table.add_column("Category")
    table.add_column("Total Spent", justify="right")
    table.add_column("Visits", justify="right")

    for m in data:
        table.add_row(
            m["name"].title(),
            (m["category"] or "unknown").replace("_", " ").title(),
            f"[expense]${m['total_spent']:.2f}[/expense]",
            str(m["transaction_count"]),
        )
    console.print(table)


@app.command()
def export(
    run_id: Annotated[str, typer.Argument(help="Run ID to export (from `receipt history`)")],
) -> None:
    """Export a past analysis as a Markdown report."""
    _load_env()
    from receipt.storage.store import ReceiptStore

    store = ReceiptStore()
    run = store.get_analysis_run(run_id)

    if not run:
        console.print(f"[expense]Run ID not found: {run_id}[/expense]")
        raise typer.Exit(1)

    lines = [f"# Receipt Analysis: {run_id}", ""]
    lines.append(f"**Period:** {run['period_start'][:10]} – {run['period_end'][:10]}")
    lines.append(f"**Transactions:** {run['transaction_count']}")
    lines.append(f"**Source:** {run.get('source_file', 'unknown')}")
    lines.append("")

    narrative = run.get("narrative")
    if narrative:
        lines.append(f"## TL;DR\n\n{narrative.get('tldr', '')}\n")
        lines.append("## Insights")
        for insight in narrative.get("insights", []):
            lines.append(f"\n### {insight['headline']}\n\n{insight['detail']}")
        lines.append(f"\n## Next Steps\n\n{narrative.get('next_steps', '')}")
    else:
        lines.append("_No narrative available for this run._")

    typer.echo("\n".join(lines))


@app.command()
def serve(
    port: Annotated[int, typer.Option("--port", help="Port to listen on")] = 8000,
    host: Annotated[str, typer.Option("--host", help="Host to bind")] = "0.0.0.0",
) -> None:
    """Start the FastAPI REST server."""
    try:
        import uvicorn
    except ImportError:
        console.print("[expense]uvicorn not installed. Run: pip install uvicorn[/expense]")
        raise typer.Exit(1)

    console.print(f"[info]Starting receipt API on http://{host}:{port}[/info]")
    uvicorn.run("receipt.api.server:app", host=host, port=port, reload=False)
