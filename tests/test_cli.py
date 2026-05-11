"""CLI tests using Typer's CliRunner with mocked pipeline internals."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from receipt.cli import app

runner = CliRunner()

# ---------------------------------------------------------------------------
# Shared fake data
# ---------------------------------------------------------------------------

_FAKE_DF = pd.DataFrame(
    {
        "date": pd.to_datetime(["2026-04-01", "2026-04-10", "2026-04-20"]).tz_localize("UTC"),
        "description": ["Starbucks", "Netflix", "Payroll"],
        "amount": [-5.50, -15.99, 3000.0],
        "raw_description": ["STARBUCKS", "NETFLIX.COM", "PAYROLL DEPOSIT"],
        "source": ["chase"] * 3,
        "transaction_id": [f"t{i:013d}" for i in range(3)],
        "category": ["food_drink", "entertainment", "income"],
        "is_anomaly": [False, False, False],
        "merchant": ["Starbucks", "Netflix", "Employer"],
    }
)

_FAKE_STATS = {
    "total_spent": -21.49,
    "total_income": 3000.0,
    "net": 2978.51,
    "subscription_total": 15.99,
    "by_category": {
        "food_drink": {"total": 5.50, "count": 1, "avg": 5.50},
        "entertainment": {"total": 15.99, "count": 1, "avg": 15.99},
    },
    "most_frequent_merchant": {"merchant": "Starbucks", "count": 1},
    "largest_single_transaction": {"amount": -15.99, "description": "Netflix"},
}

_FAKE_PATTERNS = [
    SimpleNamespace(type="subscription", headline="Netflix recurring", severity="info", data={}),
]

_FAKE_NARRATIVE = SimpleNamespace(
    tldr="You spent $21.49 this month.",
    insights=[SimpleNamespace(headline="Coffee habit", detail="You visit Starbucks regularly.")],
    next_steps="Consider reducing subscriptions.",
    to_dict=lambda self=None: {
        "tldr": "You spent $21.49 this month.",
        "insights": [{"headline": "Coffee habit", "detail": "You visit Starbucks regularly."}],
        "next_steps": "Consider reducing subscriptions.",
    },
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_pipeline(mocker, *, narrative=None, drift=None, save_run_id=None):
    """Patch all heavy pipeline/storage internals and return the mock namespace."""
    ns = SimpleNamespace()

    # Parser / ingestion
    ns.parser = mocker.MagicMock()
    ns.parser.parse.return_value = _FAKE_DF.copy()
    mocker.patch("receipt.ingestion.detect_parser", return_value=ns.parser)

    # Cleaner
    mocker.patch("receipt.pipeline.cleaner.normalize_descriptions", side_effect=lambda df: df)
    mocker.patch("receipt.pipeline.cleaner.deduplicate", side_effect=lambda df: df)
    mocker.patch("receipt.pipeline.cleaner.normalize_dates", side_effect=lambda df: df)

    # Categorizer
    ns.categorizer = mocker.MagicMock()
    ns.categorizer.categorize.side_effect = lambda df: df
    mocker.patch("receipt.pipeline.categorizer.SemanticCategorizer", return_value=ns.categorizer)

    # Anomaly detector
    ns.anomaly = mocker.MagicMock()
    ns.anomaly.fit_predict.side_effect = lambda df: df
    mocker.patch("receipt.analysis.anomalies.AnomalyDetector", return_value=ns.anomaly)

    # Aggregator
    mocker.patch("receipt.pipeline.aggregator.compute_stats", return_value=_FAKE_STATS)

    # Patterns
    mocker.patch("receipt.analysis.patterns.detect_patterns", return_value=_FAKE_PATTERNS)

    # Drift
    if drift is not None:
        mocker.patch("receipt.storage.store.ReceiptStore", return_value=mocker.MagicMock())
        mocker.patch("receipt.pipeline.drift.DriftDetector", return_value=mocker.MagicMock())

    # Narrator
    if narrative is not None:
        ns.narrator = mocker.MagicMock()
        ns.narrator.generate_narrative.return_value = narrative
        mocker.patch("receipt.analysis.narrator.Narrator", return_value=ns.narrator)

    # Storage
    store_mock = mocker.MagicMock()
    store_mock.save_analysis.return_value = save_run_id or "run-abc-123"
    mocker.patch("receipt.storage.store.ReceiptStore", return_value=store_mock)
    ns.store = store_mock

    return ns


# ---------------------------------------------------------------------------
# analyze — terminal output (default)
# ---------------------------------------------------------------------------

class TestAnalyzeTerminal:
    def test_uses_sample_when_no_file_given(self, mocker, tmp_path):
        _patch_pipeline(mocker)
        sample = tmp_path / "chase_sample.csv"
        sample.write_text("Transaction Date,Description,Amount\n2026-04-01,Test,-5.00\n")
        mocker.patch("receipt.cli.Path.__truediv__", return_value=sample)

        result = runner.invoke(app, ["analyze"])
        assert result.exit_code == 0

    def test_analyze_with_explicit_file(self, mocker, tmp_path):
        _patch_pipeline(mocker)
        csv = tmp_path / "test.csv"
        csv.write_text("date,description,amount\n2026-04-01,Test,-5.00\n")

        result = runner.invoke(app, ["analyze", str(csv), "--period", "9999"])
        assert result.exit_code == 0
        assert "Spending Summary" in result.output or "receipt analysis" in result.output.lower()

    def test_shows_category_table(self, mocker, tmp_path):
        _patch_pipeline(mocker)
        csv = tmp_path / "test.csv"
        csv.write_text("date,description,amount\n2026-04-01,Test,-5.00\n")

        result = runner.invoke(app, ["analyze", str(csv), "--period", "9999"])
        assert result.exit_code == 0
        assert "By Category" in result.output or "food" in result.output.lower()

    def test_unknown_format_exits_1(self, tmp_path):
        csv = tmp_path / "test.csv"
        csv.write_text("date,description,amount\n2026-04-01,Test,-5.00\n")

        result = runner.invoke(app, ["analyze", str(csv), "--format", "unknown"])
        assert result.exit_code == 1

    def test_missing_file_exits_1(self):
        result = runner.invoke(app, ["analyze", "/nonexistent/path/data.csv"])
        assert result.exit_code == 1

    def test_narrative_section_shown_when_api_key_present(self, mocker, tmp_path):
        _patch_pipeline(mocker, narrative=_FAKE_NARRATIVE)
        csv = tmp_path / "test.csv"
        csv.write_text("date,description,amount\n2026-04-01,Test,-5.00\n")

        result = runner.invoke(
            app,
            ["analyze", str(csv), "--period", "9999", "--api-key", "sk-fake"],
        )
        assert result.exit_code == 0
        assert "AI Insights" in result.output or "TL;DR" in result.output


# ---------------------------------------------------------------------------
# analyze — JSON output
# ---------------------------------------------------------------------------

class TestAnalyzeJSON:
    def test_json_output_is_valid(self, mocker, tmp_path):
        _patch_pipeline(mocker)
        csv = tmp_path / "test.csv"
        csv.write_text("date,description,amount\n2026-04-01,Test,-5.00\n")

        result = runner.invoke(
            app, ["analyze", str(csv), "--period", "9999", "--output", "json"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "stats" in data
        assert "patterns" in data

    def test_json_stats_keys(self, mocker, tmp_path):
        _patch_pipeline(mocker)
        csv = tmp_path / "test.csv"
        csv.write_text("date,description,amount\n2026-04-01,Test,-5.00\n")

        result = runner.invoke(
            app, ["analyze", str(csv), "--period", "9999", "--output", "json"]
        )
        data = json.loads(result.output)
        for key in ("total_spent", "total_income", "net"):
            assert key in data["stats"]

    def test_json_patterns_list(self, mocker, tmp_path):
        _patch_pipeline(mocker)
        csv = tmp_path / "test.csv"
        csv.write_text("date,description,amount\n2026-04-01,Test,-5.00\n")

        result = runner.invoke(
            app, ["analyze", str(csv), "--period", "9999", "--output", "json"]
        )
        data = json.loads(result.output)
        assert isinstance(data["patterns"], list)
        assert data["patterns"][0]["type"] == "subscription"

    def test_json_with_narrative(self, mocker, tmp_path):
        _patch_pipeline(mocker, narrative=_FAKE_NARRATIVE)
        csv = tmp_path / "test.csv"
        csv.write_text("date,description,amount\n2026-04-01,Test,-5.00\n")

        result = runner.invoke(
            app,
            ["analyze", str(csv), "--period", "9999", "--output", "json", "--api-key", "sk-fake"],
        )
        data = json.loads(result.output)
        assert data["narrative"] is not None
        assert "tldr" in data["narrative"]


# ---------------------------------------------------------------------------
# analyze — Markdown output
# ---------------------------------------------------------------------------

class TestAnalyzeMarkdown:
    def test_markdown_output_has_heading(self, mocker, tmp_path):
        _patch_pipeline(mocker)
        csv = tmp_path / "test.csv"
        csv.write_text("date,description,amount\n2026-04-01,Test,-5.00\n")

        result = runner.invoke(
            app, ["analyze", str(csv), "--period", "9999", "--output", "markdown"]
        )
        assert result.exit_code == 0
        assert "# Receipt Analysis Report" in result.output

    def test_markdown_includes_summary(self, mocker, tmp_path):
        _patch_pipeline(mocker)
        csv = tmp_path / "test.csv"
        csv.write_text("date,description,amount\n2026-04-01,Test,-5.00\n")

        result = runner.invoke(
            app, ["analyze", str(csv), "--period", "9999", "--output", "markdown"]
        )
        assert "Total Spent" in result.output
        assert "By Category" in result.output


# ---------------------------------------------------------------------------
# analyze — --save flag
# ---------------------------------------------------------------------------

class TestAnalyzeSave:
    def test_save_calls_store(self, mocker, tmp_path):
        ns = _patch_pipeline(mocker, save_run_id="run-999")
        csv = tmp_path / "test.csv"
        csv.write_text("date,description,amount\n2026-04-01,Test,-5.00\n")

        result = runner.invoke(
            app, ["analyze", str(csv), "--period", "9999", "--save"]
        )
        assert result.exit_code == 0
        ns.store.save_analysis.assert_called_once()
        ns.store.save_transactions.assert_called_once()
        ns.store.upsert_merchants.assert_called_once()

    def test_run_id_shown_in_output(self, mocker, tmp_path):
        _patch_pipeline(mocker, save_run_id="run-xyz")
        csv = tmp_path / "test.csv"
        csv.write_text("date,description,amount\n2026-04-01,Test,-5.00\n")

        result = runner.invoke(
            app, ["analyze", str(csv), "--period", "9999", "--save"]
        )
        assert "run-xyz" in result.output


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------

class TestHistory:
    def test_history_shows_table(self, mocker):
        store = mocker.MagicMock()
        store.get_analysis_history.return_value = [
            {
                "run_id": "run-abc",
                "created_at": "2026-04-20T10:00:00",
                "period_start": "2026-03-21",
                "period_end": "2026-04-20",
                "transaction_count": 42,
                "tldr": "You spent a lot.",
            }
        ]
        mocker.patch("receipt.storage.store.ReceiptStore", return_value=store)

        result = runner.invoke(app, ["history"])
        assert result.exit_code == 0
        assert "run-abc" in result.output
        assert "Analysis History" in result.output

    def test_history_empty_message(self, mocker):
        store = mocker.MagicMock()
        store.get_analysis_history.return_value = []
        mocker.patch("receipt.storage.store.ReceiptStore", return_value=store)

        result = runner.invoke(app, ["history"])
        assert result.exit_code == 0
        assert "No analysis runs found" in result.output


# ---------------------------------------------------------------------------
# merchants
# ---------------------------------------------------------------------------

class TestMerchants:
    def test_merchants_shows_table(self, mocker):
        store = mocker.MagicMock()
        store.get_merchants.return_value = [
            {"name": "starbucks", "category": "food_drink", "total_spent": 45.0, "transaction_count": 9},
        ]
        mocker.patch("receipt.storage.store.ReceiptStore", return_value=store)

        result = runner.invoke(app, ["merchants"])
        assert result.exit_code == 0
        assert "Starbucks" in result.output
        assert "Top Merchants" in result.output

    def test_merchants_empty_message(self, mocker):
        store = mocker.MagicMock()
        store.get_merchants.return_value = []
        mocker.patch("receipt.storage.store.ReceiptStore", return_value=store)

        result = runner.invoke(app, ["merchants"])
        assert result.exit_code == 0
        assert "No merchant data found" in result.output


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_prints_markdown(self, mocker):
        store = mocker.MagicMock()
        store.get_analysis_run.return_value = {
            "run_id": "run-abc",
            "period_start": "2026-03-21",
            "period_end": "2026-04-20",
            "transaction_count": 10,
            "source_file": "test.csv",
            "narrative": {
                "tldr": "You did fine.",
                "insights": [{"headline": "Savings up", "detail": "Well done."}],
                "next_steps": "Keep going.",
            },
        }
        mocker.patch("receipt.storage.store.ReceiptStore", return_value=store)

        result = runner.invoke(app, ["export", "run-abc"])
        assert result.exit_code == 0
        assert "# Receipt Analysis: run-abc" in result.output
        assert "You did fine." in result.output

    def test_export_missing_run_exits_1(self, mocker):
        store = mocker.MagicMock()
        store.get_analysis_run.return_value = None
        mocker.patch("receipt.storage.store.ReceiptStore", return_value=store)

        result = runner.invoke(app, ["export", "nonexistent"])
        assert result.exit_code == 1

    def test_export_no_narrative(self, mocker):
        store = mocker.MagicMock()
        store.get_analysis_run.return_value = {
            "run_id": "run-abc",
            "period_start": "2026-03-21",
            "period_end": "2026-04-20",
            "transaction_count": 5,
            "source_file": "x.csv",
            "narrative": None,
        }
        mocker.patch("receipt.storage.store.ReceiptStore", return_value=store)

        result = runner.invoke(app, ["export", "run-abc"])
        assert result.exit_code == 0
        assert "No narrative available" in result.output


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

class TestServe:
    def test_serve_fails_without_uvicorn(self, mocker):
        mocker.patch.dict("sys.modules", {"uvicorn": None})
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 1

    def test_serve_starts_with_uvicorn(self, mocker):
        uvicorn_mock = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"uvicorn": uvicorn_mock})
        result = runner.invoke(app, ["serve", "--port", "9000"])
        assert result.exit_code == 0
        uvicorn_mock.run.assert_called_once_with(
            "receipt.api.server:app", host="0.0.0.0", port=9000, reload=False
        )
