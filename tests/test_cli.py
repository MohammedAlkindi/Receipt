"""CLI tests using Typer's CliRunner with mocked pipeline internals."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
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
    """Patch all heavy pipeline/storage internals and return the mock namespace.

    ReceiptStore is patched exactly once at the top so all code paths
    (including drift) share the same mock instance.
    """
    ns = SimpleNamespace()

    # Storage — single patch, used by all branches
    store_mock = mocker.MagicMock()
    store_mock.save_analysis.return_value = save_run_id or "run-abc-123"
    mocker.patch("receipt.storage.store.ReceiptStore", return_value=store_mock)
    ns.store = store_mock

    # Parser / ingestion
    ns.parser = mocker.MagicMock()
    ns.parser.parse.return_value = _FAKE_DF.copy()
    mocker.patch("receipt.ingestion.detect_parser", return_value=ns.parser)

    # Cleaner
    mocker.patch("receipt.pipeline.cleaner.normalize_descriptions", side_effect=lambda df, **kwargs: df)
    mocker.patch("receipt.pipeline.cleaner.deduplicate", side_effect=lambda df, **kwargs: df)
    mocker.patch("receipt.pipeline.cleaner.normalize_dates", side_effect=lambda df, **kwargs: df)

    # Categorizer
    ns.categorizer = mocker.MagicMock()
    ns.categorizer.categorize.side_effect = lambda df, **kwargs: df
    mocker.patch("receipt.pipeline.categorizer.SemanticCategorizer", return_value=ns.categorizer)

    # Anomaly detector
    ns.anomaly = mocker.MagicMock()
    ns.anomaly.fit_predict.side_effect = lambda df, **kwargs: df
    mocker.patch("receipt.analysis.anomalies.AnomalyDetector", return_value=ns.anomaly)

    # Aggregator
    mocker.patch("receipt.pipeline.aggregator.compute_stats", return_value=_FAKE_STATS)

    # Patterns
    mocker.patch("receipt.analysis.patterns.detect_patterns", return_value=_FAKE_PATTERNS)

    # Drift
    if drift is not None:
        mocker.patch("receipt.pipeline.drift.DriftDetector", return_value=mocker.MagicMock())

    # Narrator
    if narrative is not None:
        ns.narrator = mocker.MagicMock()
        ns.narrator.generate_narrative.return_value = narrative
        mocker.patch("receipt.analysis.narrator.Narrator", return_value=ns.narrator)

    return ns


# ---------------------------------------------------------------------------
# analyze — terminal output (default)
# ---------------------------------------------------------------------------

class TestAnalyzeTerminal:
    def test_analyze_requires_file_argument(self):
        result = runner.invoke(app, ["analyze"])
        assert result.exit_code != 0

    def test_analyze_missing_file_exits_1(self):
        result = runner.invoke(app, ["analyze", "/nonexistent/path/data.csv"])
        assert result.exit_code == 1

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

    def test_out_of_range_period_exits_1(self, tmp_path):
        """M3 regression: the CLI rejects a period outside 1..36500 instead of
        letting timedelta overflow."""
        csv = tmp_path / "test.csv"
        csv.write_text("date,description,amount\n2026-04-01,Test,-5.00\n")

        result = runner.invoke(app, ["analyze", str(csv), "--period", "1000000000"])
        assert result.exit_code == 1
        assert "period" in result.output.lower()

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
        # M4: anomalies are part of the JSON contract (empty here, key present).
        assert "anomalies" in data
        assert isinstance(data["anomalies"], list)

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
# FastAPI endpoint typing tests (Task 10)
# ---------------------------------------------------------------------------

class TestAPIResponseModels:
    def test_health_response_typed(self, mocker, tmp_path):
        """Task 10: /health returns JSON with status, db, version, narrative_service keys."""
        from fastapi.testclient import TestClient

        from receipt.api.server import app as fastapi_app

        mocker.patch(
            "receipt.storage.store.ReceiptStore",
            return_value=mocker.MagicMock(
                db_stats=mocker.MagicMock(
                    return_value={"transactions": 0, "analysis_runs": 0, "merchants": 0}
                )
            ),
        )

        client = TestClient(fastapi_app)
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert "db" in body
        assert "version" in body
        assert "narrative_service" in body

    def test_history_endpoint_returns_typed_list(self, mocker):
        """Task 10: /history returns list of AnalysisRunSummary-shaped dicts."""
        from fastapi.testclient import TestClient

        from receipt.api.server import app as fastapi_app

        mocker.patch(
            "receipt.storage.store.ReceiptStore",
            return_value=mocker.MagicMock(
                get_analysis_history=mocker.MagicMock(
                    return_value=[
                        {
                            "run_id": "r1",
                            "created_at": "2026-04-01T00:00:00",
                            "period_start": "2026-03-01T00:00:00",
                            "period_end": "2026-03-31T00:00:00",
                            "source_file": "test.csv",
                            "transaction_count": 10,
                            "tldr": "Good month.",
                        }
                    ]
                )
            ),
        )

        client = TestClient(fastapi_app)
        resp = client.get("/history")
        assert resp.status_code == 200
        items = resp.json()
        assert isinstance(items, list)
        assert items[0]["run_id"] == "r1"


class TestAPISecurity:
    def test_token_enforced_when_configured(self, mocker, monkeypatch):
        """H5 regression: with RECEIPT_API_TOKEN set, data routes reject a
        request that lacks (or mismatches) the X-Receipt-Token header."""
        from fastapi.testclient import TestClient

        from receipt.api.server import app as fastapi_app

        monkeypatch.setenv("RECEIPT_API_TOKEN", "s3cret")
        mocker.patch(
            "receipt.storage.store.ReceiptStore",
            return_value=mocker.MagicMock(
                get_analysis_history=mocker.MagicMock(return_value=[])
            ),
        )
        client = TestClient(fastapi_app)

        assert client.get("/history").status_code == 401
        assert client.get("/history", headers={"X-Receipt-Token": "wrong"}).status_code == 401
        assert client.get("/history", headers={"X-Receipt-Token": "s3cret"}).status_code == 200

    def test_health_open_without_token(self, mocker, monkeypatch):
        """/health stays open as a liveness probe even when a token is set."""
        from fastapi.testclient import TestClient

        from receipt.api.server import app as fastapi_app

        monkeypatch.setenv("RECEIPT_API_TOKEN", "s3cret")
        mocker.patch(
            "receipt.storage.store.ReceiptStore",
            return_value=mocker.MagicMock(
                db_stats=mocker.MagicMock(
                    return_value={"transactions": 0, "analysis_runs": 0, "merchants": 0}
                )
            ),
        )
        client = TestClient(fastapi_app)
        assert client.get("/health").status_code == 200

    def test_body_cap_counts_streamed_bytes(self, monkeypatch):
        """H5 regression: an oversized body with NO Content-Length header
        (chunked transfer) is still rejected with 413."""
        from fastapi.testclient import TestClient

        from receipt.api.server import app as fastapi_app

        monkeypatch.delenv("RECEIPT_API_TOKEN", raising=False)
        client = TestClient(fastapi_app)

        def chunked_body():
            # 11 MB streamed in 1 MB chunks; httpx omits Content-Length for a
            # generator body, forcing chunked transfer encoding.
            for _ in range(11):
                yield b"x" * (1024 * 1024)

        resp = client.post(
            "/analyze",
            content=chunked_body(),
            headers={"X-Api-Key": "sk-ant-test", "Content-Type": "application/json"},
        )
        assert resp.status_code == 413


class TestAnalyzeEndpoint:
    """Integration tests for POST /analyze (keyword categorizer, mocked API)."""

    def _b64(self, text: str) -> str:
        import base64

        return base64.b64encode(text.encode()).decode()

    def _mock_pipeline(self, mocker, tmp_path, monkeypatch):
        """Force keyword categorization (no model download) and mock the
        Anthropic call; route the store at a temp DB."""
        from receipt.analysis.narrator import NarrativeReport
        from receipt.pipeline.categorizer import SemanticCategorizer as RealCat

        monkeypatch.setenv("RECEIPT_DB_PATH", str(tmp_path / "api.db"))
        monkeypatch.delenv("RECEIPT_API_TOKEN", raising=False)

        mocker.patch(
            "receipt.pipeline.categorizer.SemanticCategorizer",
            side_effect=lambda *a, **k: RealCat(use_embeddings=False),
        )
        mock_narrator = mocker.MagicMock()
        mock_narrator.generate_narrative.return_value = NarrativeReport(
            tldr="t", insights=[], next_steps="n"
        )
        mocker.patch("receipt.analysis.narrator.Narrator", return_value=mock_narrator)
        return mock_narrator

    def test_period_out_of_range_returns_422(self, monkeypatch):
        """M3 regression: period=1e9 is a typed 422, not a raw 500."""
        from fastapi.testclient import TestClient

        from receipt.api.server import app as fastapi_app

        monkeypatch.delenv("RECEIPT_API_TOKEN", raising=False)
        client = TestClient(fastapi_app)
        resp = client.post(
            "/analyze",
            json={"file_content": self._b64("date,description,amount\n2026-07-20,X,-5\n"),
                  "format": "generic", "period": 10**9},
            headers={"X-Api-Key": "sk-ant-test"},
        )
        assert resp.status_code == 422

    def test_dedup_window_out_of_range_returns_422(self, monkeypatch):
        """M3 regression: dedup_window_days=9 is a typed 422, not a raw 500."""
        from fastapi.testclient import TestClient

        from receipt.api.server import app as fastapi_app

        monkeypatch.delenv("RECEIPT_API_TOKEN", raising=False)
        client = TestClient(fastapi_app)
        resp = client.post(
            "/analyze",
            json={"file_content": self._b64("date,description,amount\n2026-07-20,X,-5\n"),
                  "format": "generic", "dedup_window_days": 9},
            headers={"X-Api-Key": "sk-ant-test"},
        )
        assert resp.status_code == 422

    def test_analyze_happy_path(self, mocker, tmp_path, monkeypatch):
        """/analyze returns a typed response and passes drift=None to the
        narrator when there is no prior period."""
        from fastapi.testclient import TestClient

        from receipt.api.server import app as fastapi_app

        narrator = self._mock_pipeline(mocker, tmp_path, monkeypatch)
        client = TestClient(fastapi_app)
        csv = "date,description,amount\n2026-07-20,Starbucks,-5.00\n2026-07-21,Netflix,-15.49\n"
        resp = client.post(
            "/analyze",
            json={"file_content": self._b64(csv), "format": "generic", "period": 60},
            headers={"X-Api-Key": "sk-ant-test"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["drift"] is None
        assert body["transaction_count"] == 2
        assert "anomalies" in body and isinstance(body["anomalies"], list)
        # narrator received the drift positional arg (None here)
        args, _ = narrator.generate_narrative.call_args
        assert args[2] is None

    def test_analyze_computes_drift_against_prior_period(self, mocker, tmp_path, monkeypatch):
        """M3 regression: the API computes drift vs the previous stored period
        (previously it always returned drift=None)."""
        from fastapi.testclient import TestClient

        from receipt.api.server import app as fastapi_app
        from receipt.storage.store import ReceiptStore

        narrator = self._mock_pipeline(mocker, tmp_path, monkeypatch)

        # Seed a prior (June) run with categorized transactions.
        store = ReceiptStore()
        prior = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-06-10", "2026-06-12"]).tz_localize("UTC"),
                "description": ["Uber Eats", "Netflix"],
                "raw_description": ["Uber Eats", "Netflix"],
                "amount": [-20.0, -15.49],
                "source": ["generic", "generic"],
                "transaction_id": ["prior1", "prior2"],
                "category": ["food_dining", "subscriptions"],
            }
        )
        run_id = store.save_analysis(
            period_start=prior["date"].min().to_pydatetime(),
            period_end=prior["date"].max().to_pydatetime(),
            transaction_count=len(prior),
        )
        store.save_transactions(prior, run_id)

        client = TestClient(fastapi_app)
        csv = (
            "date,description,amount\n"
            "2026-07-18,Uber Eats,-60.00\n"
            "2026-07-20,Netflix,-15.49\n"
        )
        resp = client.post(
            "/analyze",
            json={"file_content": self._b64(csv), "format": "generic", "period": 40},
            headers={"X-Api-Key": "sk-ant-test"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["drift"] is not None
        assert "velocity_trend" in body["drift"]
        # narrator received a non-None drift arg
        args, _ = narrator.generate_narrative.call_args
        assert args[2] is not None


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

class TestModuleEntryPoint:
    def test_python_m_receipt_cli_runs(self):
        """M6 regression: `python -m receipt.cli` must execute the app, not
        silently no-op (the README documents it as a fallback invocation)."""
        import os
        import subprocess
        import sys

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            [sys.executable, "-m", "receipt.cli", "--help"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=60,
        )
        assert result.returncode == 0
        assert result.stdout.strip(), "python -m receipt.cli produced no output"
        assert "analyze" in result.stdout


class TestServe:
    def test_serve_starts_with_uvicorn(self, mocker):
        uvicorn_mock = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"uvicorn": uvicorn_mock})
        result = runner.invoke(app, ["serve", "--port", "9000"])
        assert result.exit_code == 0
        uvicorn_mock.run.assert_called_once_with(
            "receipt.api.server:app", host="127.0.0.1", port=9000, reload=False
        )

    def test_serve_defaults_to_loopback(self, mocker, monkeypatch):
        """H5 regression: serve must bind loopback by default, not 0.0.0.0."""
        monkeypatch.delenv("RECEIPT_API_HOST", raising=False)
        monkeypatch.delenv("RECEIPT_API_PORT", raising=False)
        uvicorn_mock = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"uvicorn": uvicorn_mock})
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0
        _, kwargs = uvicorn_mock.run.call_args
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 8000

    def test_serve_honors_env_host_port(self, mocker, monkeypatch):
        """H5 regression: RECEIPT_API_HOST/PORT override the defaults."""
        monkeypatch.setenv("RECEIPT_API_HOST", "127.0.0.1")
        monkeypatch.setenv("RECEIPT_API_PORT", "7777")
        uvicorn_mock = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"uvicorn": uvicorn_mock})
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0
        _, kwargs = uvicorn_mock.run.call_args
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 7777
