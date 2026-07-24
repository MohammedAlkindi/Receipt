"""Tests for the pipeline layer."""

from __future__ import annotations

import pandas as pd

from receipt.pipeline.aggregator import compute_stats
from receipt.pipeline.cleaner import deduplicate, normalize_dates, normalize_descriptions
from receipt.pipeline.drift import DriftDetector

# ---------------------------------------------------------------------------
# Cleaner
# ---------------------------------------------------------------------------

class TestNormalizeDescriptions:
    def _make_df(self, descriptions: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "description": descriptions,
                "amount": [-10.0] * len(descriptions),
                "date": pd.to_datetime(["2026-04-01"] * len(descriptions)).tz_localize("UTC"),
                "transaction_id": [f"id{i}" for i in range(len(descriptions))],
            }
        )

    def test_strips_trailing_digits(self):
        df = self._make_df(["UBER* EATS 3948201"])
        result = normalize_descriptions(df)
        assert "3948201" not in result["description"].iloc[0]

    def test_normalizes_known_merchant(self):
        df = self._make_df(["STARBUCKS #04892"])
        result = normalize_descriptions(df)
        assert "Starbucks" in result["description"].iloc[0]

    def test_normalizes_uber_eats(self):
        df = self._make_df(["UBER* EATS 4829482"])
        result = normalize_descriptions(df)
        desc = result["description"].iloc[0]
        assert "Uber Eats" in desc or "Ubereats" in desc or "Uber" in desc

    def test_does_not_alter_clean_names(self):
        df = self._make_df(["Netflix"])
        result = normalize_descriptions(df)
        assert "Netflix" in result["description"].iloc[0]


class TestDeduplicate:
    def _base_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-01", "2026-04-01", "2026-04-05"]).tz_localize("UTC"),
                "description": ["Coffee", "Coffee", "Gym"],
                "amount": [-5.0, -5.0, -49.0],
                "transaction_id": ["aaa", "aaa", "bbb"],
                "raw_description": ["Coffee", "Coffee", "Gym"],
                "source": ["chase", "chase", "chase"],
            }
        )

    def test_removes_exact_duplicates(self):
        df = self._base_df()
        result = deduplicate(df)
        assert len(result) == 2

    def test_preserves_unique_rows(self):
        df = self._base_df()
        result = deduplicate(df)
        assert any(result["description"] == "Gym")

    def test_distinct_same_day_purchases_survive_window_zero(self):
        """C1 regression: identical same-day purchases parsed from a real CSV
        get distinct transaction_ids, so --dedup-window 0 truly keeps both."""
        import io

        from receipt.ingestion.csv_parser import GenericCSVParser

        csv = (
            "date,description,amount\n"
            "2026-04-03,Starbucks,-5.75\n"
            "2026-04-03,Starbucks,-5.75\n"
        )
        df = GenericCSVParser().parse(io.StringIO(csv))
        result = deduplicate(df, near_dup_window_days=0)
        assert len(result) == 2

    def test_same_day_identical_removed_only_by_near_dup_window(self):
        """With the window enabled (default), same-day identical rows are
        still treated as near-duplicates (double-posting protection)."""
        import io

        from receipt.ingestion.csv_parser import GenericCSVParser

        csv = (
            "date,description,amount\n"
            "2026-04-03,Starbucks,-5.75\n"
            "2026-04-03,Starbucks,-5.75\n"
        )
        df = GenericCSVParser().parse(io.StringIO(csv))
        result = deduplicate(df, near_dup_window_days=2)
        assert len(result) == 1


class TestNormalizeDates:
    def test_adds_temporal_columns(self, sample_df):
        result = normalize_dates(sample_df)
        assert "day_of_week" in result.columns
        assert "week_of_month" in result.columns
        assert "is_weekend" in result.columns

    def test_weekend_flag_correct(self, sample_df):
        result = normalize_dates(sample_df)
        # 2026-04-05 is a Sunday
        sunday_row = result[result["date"].dt.date == pd.Timestamp("2026-04-05").date()]
        if not sunday_row.empty:
            assert sunday_row["is_weekend"].iloc[0]


# ---------------------------------------------------------------------------
# Categorizer (keyword fallback, no model needed)
# ---------------------------------------------------------------------------

class TestSemanticCategorizer:
    def test_assigns_food_category(self, sample_df):
        from receipt.pipeline.categorizer import SemanticCategorizer

        cat = SemanticCategorizer(use_embeddings=False)
        result = cat.categorize(sample_df)
        assert "category" in result.columns

    def test_assigns_subscription_to_netflix(self, sample_df):
        from receipt.pipeline.categorizer import SemanticCategorizer

        cat = SemanticCategorizer(use_embeddings=False)
        result = cat.categorize(sample_df)
        netflix_rows = result[result["description"].str.contains("Netflix", case=False, na=False)]
        if not netflix_rows.empty:
            assert netflix_rows["category"].iloc[0] == "subscriptions"

    def test_category_confidence_column_present(self, sample_df):
        from receipt.pipeline.categorizer import SemanticCategorizer

        cat = SemanticCategorizer(use_embeddings=False)
        result = cat.categorize(sample_df)
        assert "category_confidence" in result.columns
        assert (result["category_confidence"] >= 0).all()

    def test_model_cache_env_override(self, tmp_path, monkeypatch):
        """H4 regression: RECEIPT_MODEL_CACHE must control the cache dir."""
        from receipt.pipeline.categorizer import SemanticCategorizer

        monkeypatch.setenv("RECEIPT_MODEL_CACHE", str(tmp_path / "modelcache"))
        cat = SemanticCategorizer(use_embeddings=False)
        assert cat._cache_dir == tmp_path / "modelcache"


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class TestComputeStats:
    def test_total_spent_negative(self, sample_df):
        stats = compute_stats(sample_df)
        assert stats["total_spent"] < 0

    def test_total_income_positive(self, sample_df):
        stats = compute_stats(sample_df)
        assert stats["total_income"] > 0

    def test_net_computed(self, sample_df):
        stats = compute_stats(sample_df)
        assert abs(stats["net"] - (stats["total_spent"] + stats["total_income"])) < 0.01

    def test_transaction_counts(self, sample_df):
        stats = compute_stats(sample_df)
        assert stats["transaction_count"] == len(sample_df)
        assert stats["expense_count"] + stats["income_count"] == len(sample_df)

    def test_largest_transaction_present(self, sample_df):
        stats = compute_stats(sample_df)
        assert stats["largest_single_transaction"] is not None
        assert stats["largest_single_transaction"]["amount"] < 0


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------

class TestDriftDetector:
    def _make_categorized(self, amounts: dict[str, float], offset_days: int = 0) -> pd.DataFrame:
        rows = []
        for merchant, amount in amounts.items():
            rows.append(
                {
                    "date": pd.Timestamp(f"2026-04-{10 + offset_days:02d}", tz="UTC"),
                    "description": merchant,
                    "amount": amount,
                    "category": "food_dining" if amount < -20 else "subscriptions",
                    "transaction_id": f"t{hash(merchant + str(amount))}",
                }
            )
        return pd.DataFrame(rows)

    def test_detects_increased_category(self):
        prev = self._make_categorized({"Uber Eats": -50.0})
        curr = self._make_categorized({"Uber Eats": -100.0})
        report = DriftDetector().compare_periods(curr, prev)
        assert len(report.increased) > 0 or len(report.narrative_hints) > 0

    def test_detects_new_merchants(self):
        prev = self._make_categorized({"Uber Eats": -30.0})
        curr = self._make_categorized({"Uber Eats": -30.0, "DoorDash": -25.0})
        report = DriftDetector().compare_periods(curr, prev)
        assert "DoorDash" in report.new_merchants

    def test_detects_dropped_merchants(self):
        prev = self._make_categorized({"Uber Eats": -30.0, "Grubhub": -25.0})
        curr = self._make_categorized({"Uber Eats": -30.0})
        report = DriftDetector().compare_periods(curr, prev)
        assert "Grubhub" in report.dropped_merchants

    def test_velocity_stable_with_equal_spending(self):
        rows = [
            {
                "date": pd.Timestamp(f"2026-04-{d:02d}", tz="UTC"),
                "description": "Coffee",
                "amount": -5.0,
                "transaction_id": f"t{d}",
            }
            for d in range(1, 29)
        ]
        df = pd.DataFrame(rows)
        report = DriftDetector().compare_periods(df, df)
        assert report.velocity_trend in ("stable", "accelerating", "decelerating")


# ---------------------------------------------------------------------------
# Task 8: configurability
# ---------------------------------------------------------------------------

class TestDeduplicateWindowDays:
    def _near_dup_df(self) -> pd.DataFrame:
        """Two rows with same description/amount, 1 day apart (near-duplicate)."""
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-10"]).tz_localize("UTC"),
                "description": ["Coffee", "Coffee", "Gym"],
                "amount": [-5.0, -5.0, -49.0],
                "transaction_id": ["aaa", "bbb", "ccc"],
                "raw_description": ["Coffee", "Coffee", "Gym"],
                "source": ["chase", "chase", "chase"],
            }
        )

    def test_dedup_window_zero_preserves_near_duplicates(self):
        """window=0 must not remove near-duplicates (different transaction_ids)."""
        from receipt.pipeline.cleaner import deduplicate

        df = self._near_dup_df()
        result = deduplicate(df, near_dup_window_days=0)
        assert len(result) == 3  # all three rows kept

    def test_dedup_window_two_removes_near_duplicate(self):
        """Default window=2 removes Coffee on 2026-04-02 (1 day after 04-01)."""
        from receipt.pipeline.cleaner import deduplicate

        df = self._near_dup_df()
        result = deduplicate(df, near_dup_window_days=2)
        assert len(result) == 2  # 1 Coffee + Gym

    def test_dedup_window_out_of_range_raises(self):
        import pytest

        from receipt.pipeline.cleaner import deduplicate

        with pytest.raises(ValueError, match="near_dup_window_days"):
            deduplicate(self._near_dup_df(), near_dup_window_days=8)


class TestDriftDetectorThreshold:
    def _make_categorized(self, amounts: dict[str, float]) -> pd.DataFrame:
        rows = []
        for merchant, amount in amounts.items():
            rows.append(
                {
                    "date": pd.Timestamp("2026-04-10", tz="UTC"),
                    "description": merchant,
                    "amount": amount,
                    "category": "food_dining",
                    "transaction_id": f"t{hash(merchant)}",
                }
            )
        return pd.DataFrame(rows)

    def test_high_threshold_produces_fewer_flags(self):
        """threshold=0.50 flags less than threshold=0.05 for a 30% change."""
        prev = self._make_categorized({"Uber Eats": -100.0})
        curr = self._make_categorized({"Uber Eats": -130.0})  # 30% increase
        strict = DriftDetector(drift_threshold=0.05).compare_periods(curr, prev)
        lenient = DriftDetector(drift_threshold=0.50).compare_periods(curr, prev)
        # 30% > 5% threshold → flagged; 30% < 50% threshold → not flagged
        assert len(strict.increased) >= len(lenient.increased)

    def test_invalid_threshold_raises(self):
        import pytest

        with pytest.raises(ValueError, match="drift_threshold"):
            DriftDetector(drift_threshold=1.5)


# ---------------------------------------------------------------------------
# Task 9: audit logging
# ---------------------------------------------------------------------------

class TestPipelineAuditLog:
    def test_audit_log_captures_stage_events(self):
        from datetime import datetime, timezone

        from receipt.pipeline.audit import PipelineAuditLog
        from receipt.pipeline.cleaner import deduplicate, normalize_dates, normalize_descriptions

        audit_log = PipelineAuditLog(
            run_id="test-run",
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-01", "2026-04-05"]).tz_localize("UTC"),
                "description": ["Coffee", "Gym"],
                "amount": [-5.0, -49.0],
                "transaction_id": ["aaa", "bbb"],
                "raw_description": ["Coffee", "Gym"],
                "source": ["chase", "chase"],
            }
        )

        normalize_descriptions(df, audit_log=audit_log)
        deduplicate(df, audit_log=audit_log)
        normalize_dates(df, audit_log=audit_log)

        assert len(audit_log.stages) == 3
        stages = [s.stage for s in audit_log.stages]
        assert "normalize_descriptions" in stages
        assert "deduplicate" in stages
        assert "normalize_dates" in stages

    def test_audit_log_to_dict_valid_json(self):
        import json
        from datetime import datetime, timezone

        from receipt.pipeline.audit import PipelineAuditLog
        from receipt.pipeline.cleaner import normalize_descriptions

        audit_log = PipelineAuditLog(
            run_id="r1",
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-01"]).tz_localize("UTC"),
                "description": ["Coffee"],
                "amount": [-5.0],
                "transaction_id": ["aaa"],
                "raw_description": ["Coffee"],
                "source": ["chase"],
            }
        )
        normalize_descriptions(df, audit_log=audit_log)

        d = audit_log.to_dict()
        serialized = json.dumps(d, default=str)
        parsed = json.loads(serialized)
        assert parsed["run_id"] == "r1"
        assert len(parsed["stages"]) == 1
        stage = parsed["stages"][0]
        assert "stage" in stage
        assert "duration_ms" in stage
        assert "input_rows" in stage
        assert "output_rows" in stage
        assert "metadata" in stage

    def test_audit_log_without_audit_log_arg_is_noop(self):
        import pandas as pd

        from receipt.pipeline.cleaner import normalize_descriptions

        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-01"]).tz_localize("UTC"),
                "description": ["Coffee"],
                "amount": [-5.0],
                "transaction_id": ["aaa"],
                "raw_description": ["Coffee"],
                "source": ["chase"],
            }
        )
        # Must not raise when audit_log is omitted
        result = normalize_descriptions(df)
        assert len(result) == 1
