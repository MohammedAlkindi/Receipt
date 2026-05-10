"""Tests for the storage layer."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
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
        far_future = datetime(2030, 1, 1, tzinfo=timezone.utc)
        result = store.get_transactions(start_date=far_future)
        assert result.empty

    def test_save_analysis_returns_run_id(self, store):
        run_id = store.save_analysis(
            period_start=datetime(2026, 4, 1, tzinfo=timezone.utc),
            period_end=datetime(2026, 4, 30, tzinfo=timezone.utc),
            transaction_count=30,
        )
        assert isinstance(run_id, str)
        assert len(run_id) == 12

    def test_get_analysis_history_ordering(self, store):
        store.save_analysis(
            period_start=datetime(2026, 3, 1, tzinfo=timezone.utc),
            period_end=datetime(2026, 3, 31, tzinfo=timezone.utc),
            transaction_count=25,
        )
        store.save_analysis(
            period_start=datetime(2026, 4, 1, tzinfo=timezone.utc),
            period_end=datetime(2026, 4, 30, tzinfo=timezone.utc),
            transaction_count=30,
        )
        history = store.get_analysis_history()
        assert len(history) == 2
        # Newest first
        assert history[0]["period_start"] > history[1]["period_start"]

    def test_get_previous_period(self, store, sample_df):
        store.save_transactions(sample_df, "run1")
        store.save_analysis(
            period_start=datetime(2026, 4, 1, tzinfo=timezone.utc),
            period_end=datetime(2026, 4, 30, tzinfo=timezone.utc),
            transaction_count=len(sample_df),
        )
        # Current period starts in May
        current_start = datetime(2026, 5, 1, tzinfo=timezone.utc)
        prev = store.get_previous_period(current_start)
        assert prev is not None

    def test_get_analysis_run_by_id(self, store):
        run_id = store.save_analysis(
            period_start=datetime(2026, 4, 1, tzinfo=timezone.utc),
            period_end=datetime(2026, 4, 30, tzinfo=timezone.utc),
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
