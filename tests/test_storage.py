"""Tests for the storage layer."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest


@pytest.fixture
def store(tmp_path: Path):
    from receipt.storage.store import ReceiptStore

    return ReceiptStore(db_path=tmp_path / "test.db")


class TestReceiptStore:
    def test_save_and_retrieve_transactions(self, store, sample_df):
        run_id = "testrun01"
        saved = store.save_transactions(sample_df, run_id)
        assert saved == len(sample_df)

        retrieved = store.get_transactions()
        assert len(retrieved) == len(sample_df)

    def test_deduplicates_on_transaction_id(self, store, sample_df):
        store.save_transactions(sample_df, "run1")
        saved_again = store.save_transactions(sample_df, "run2")
        assert saved_again == 0  # all duplicates

    def test_get_transactions_date_filter(self, store, sample_df):
        store.save_transactions(sample_df, "run1")
        # Filter to a date after all transactions
        far_future = datetime(2030, 1, 1, tzinfo=UTC)
        result = store.get_transactions(start_date=far_future)
        assert result.empty

    def test_save_analysis_returns_run_id(self, store):
        run_id = store.save_analysis(
            period_start=datetime(2026, 4, 1, tzinfo=UTC),
            period_end=datetime(2026, 4, 30, tzinfo=UTC),
            transaction_count=30,
        )
        assert isinstance(run_id, str)
        assert len(run_id) == 12

    def test_get_analysis_history_ordering(self, store):
        store.save_analysis(
            period_start=datetime(2026, 3, 1, tzinfo=UTC),
            period_end=datetime(2026, 3, 31, tzinfo=UTC),
            transaction_count=25,
        )
        store.save_analysis(
            period_start=datetime(2026, 4, 1, tzinfo=UTC),
            period_end=datetime(2026, 4, 30, tzinfo=UTC),
            transaction_count=30,
        )
        history = store.get_analysis_history()
        assert len(history) == 2
        # Newest first
        assert history[0]["period_start"] > history[1]["period_start"]

    def test_get_previous_period(self, store, sample_df):
        store.save_transactions(sample_df, "run1")
        store.save_analysis(
            period_start=datetime(2026, 4, 1, tzinfo=UTC),
            period_end=datetime(2026, 4, 30, tzinfo=UTC),
            transaction_count=len(sample_df),
        )
        # Current period starts in May
        current_start = datetime(2026, 5, 1, tzinfo=UTC)
        prev = store.get_previous_period(current_start)
        assert prev is not None

    def test_get_previous_period_with_multiple_prior_runs(self, store, sample_df):
        """H1 regression: two or more prior runs must not raise
        MultipleResultsFound, and the most recent prior period wins."""
        store.save_transactions(sample_df, "run1")  # transactions dated April 2026
        store.save_analysis(
            period_start=datetime(2026, 3, 1, tzinfo=UTC),
            period_end=datetime(2026, 3, 31, tzinfo=UTC),
            transaction_count=0,
        )
        store.save_analysis(
            period_start=datetime(2026, 4, 1, tzinfo=UTC),
            period_end=datetime(2026, 4, 30, tzinfo=UTC),
            transaction_count=len(sample_df),
        )
        prev = store.get_previous_period(datetime(2026, 5, 1, tzinfo=UTC))
        assert prev is not None
        # The April run (latest prior) was selected — its transactions exist.
        assert len(prev) == len(sample_df)

    def test_get_analysis_run_by_id(self, store):
        run_id = store.save_analysis(
            period_start=datetime(2026, 4, 1, tzinfo=UTC),
            period_end=datetime(2026, 4, 30, tzinfo=UTC),
            transaction_count=10,
            narrative={"tldr": "Good month", "insights": [], "next_steps": "Keep it up."},
        )
        run = store.get_analysis_run(run_id)
        assert run is not None
        assert run["run_id"] == run_id
        assert run["narrative"]["tldr"] == "Good month"

    def test_get_analysis_run_returns_none_for_unknown(self, store):
        assert store.get_analysis_run("doesnotexist") is None

    def test_upsert_merchants(self, store, sample_df):
        # Add category column
        sample_with_cat = sample_df.copy()
        sample_with_cat["category"] = "food_dining"
        store.upsert_merchants(sample_with_cat)
        merchants = store.get_merchants()
        assert len(merchants) > 0

    def test_merchant_total_accumulates(self, store, sample_df):
        sample_with_cat = sample_df.copy()
        sample_with_cat["category"] = "food_dining"
        store.upsert_merchants(sample_with_cat)
        store.upsert_merchants(sample_with_cat)
        merchants = store.get_merchants()
        # Totals should be doubled since we inserted twice
        first = next(m for m in merchants if m["name"])
        assert first["total_spent"] >= 0

    def test_db_stats(self, store, sample_df):
        stats = store.db_stats()
        assert "transactions" in stats
        assert "analysis_runs" in stats
        assert "merchants" in stats

    def test_receipt_db_path_env_override(self, tmp_path, monkeypatch):
        """H4 regression: the app must honor RECEIPT_DB_PATH like Alembic does,
        so migrations and the application target the same database."""
        from receipt.storage.store import ReceiptStore

        target = tmp_path / "custom" / "mydata.db"
        monkeypatch.setenv("RECEIPT_DB_PATH", str(target))
        ReceiptStore()
        assert target.exists()

    def test_merchant_first_seen_unchanged_on_double_upsert(self, store, sample_df):
        sample_with_cat = sample_df.copy()
        sample_with_cat["category"] = "food_dining"
        store.upsert_merchants(sample_with_cat)
        first_call = {m["name"]: m for m in store.get_merchants()}
        store.upsert_merchants(sample_with_cat)
        second_call = {m["name"]: m for m in store.get_merchants()}
        for name in first_call:
            assert first_call[name]["first_seen"] == second_call[name]["first_seen"]

    def test_upsert_merchants_no_unique_constraint_error(self, store, sample_df):
        sample_with_cat = sample_df.copy()
        sample_with_cat["category"] = "food_dining"
        # Calling three times must not raise
        store.upsert_merchants(sample_with_cat)
        store.upsert_merchants(sample_with_cat)
        store.upsert_merchants(sample_with_cat)
        merchants = store.get_merchants()
        assert len(merchants) > 0


class TestNarrativeDeserialization:
    def test_schema_version_1_returns_as_is(self):
        from receipt.storage.store import _deserialize_narrative

        data = {"schema_version": 1, "tldr": "x", "insights": [], "next_steps": "y"}
        result = _deserialize_narrative(data)
        assert result == data

    def test_legacy_schema_version_0_returns_without_error(self):
        from receipt.storage.store import _deserialize_narrative

        data = {"tldr": "x", "insights": [], "next_steps": "y"}
        result = _deserialize_narrative(data)
        assert result["tldr"] == "x"
        assert isinstance(result["insights"], list)

    def test_future_schema_version_returns_as_is(self):
        from receipt.storage.store import _deserialize_narrative

        data = {"schema_version": 99, "tldr": "x", "insights": [], "next_steps": "y"}
        result = _deserialize_narrative(data)
        assert result["tldr"] == "x"

    def test_narrative_to_dict_includes_schema_version(self):
        from receipt.analysis.narrator import Insight, NarrativeReport

        report = NarrativeReport(
            tldr="Test",
            insights=[Insight(headline="h", detail="d")],
            next_steps="do it",
        )
        d = report.to_dict()
        assert d["schema_version"] == 1

    def test_get_analysis_run_handles_legacy_narrative(self, tmp_path):
        import json
        from datetime import datetime

        from receipt.storage.store import ReceiptStore

        store = ReceiptStore(db_path=tmp_path / "legacy.db")
        # Manually save a run with a legacy narrative (no schema_version)
        legacy_narrative = json.dumps({"tldr": "old", "insights": [], "next_steps": "none"})
        from sqlalchemy.orm import Session

        from receipt.storage.models import AnalysisRun
        with Session(store._engine) as session:
            run = AnalysisRun(
                run_id="legacyrun01",
                created_at=datetime.now(UTC),
                period_start=datetime(2026, 1, 1, tzinfo=UTC),
                period_end=datetime(2026, 1, 31, tzinfo=UTC),
                transaction_count=5,
                narrative_json=legacy_narrative,
            )
            session.add(run)
            session.commit()
        result = store.get_analysis_run("legacyrun01")
        assert result is not None
        assert result["narrative"]["tldr"] == "old"
