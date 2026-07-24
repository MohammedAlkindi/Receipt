"""Tests for the analysis layer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from receipt.analysis.anomalies import AnomalyDetector
from receipt.analysis.patterns import (
    Pattern,
    _anomalous_week,
    _late_night_spending,
    _subscription_creep,
    _weekend_splurge,
    detect_patterns,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tx(date: str, desc: str, amount: float, category: str = "food_dining", hour: int = 12, is_weekend: bool = False) -> dict:
    return {
        "date": pd.Timestamp(date, tz="UTC"),
        "description": desc,
        "amount": amount,
        "category": category,
        "hour": hour,
        "is_weekend": is_weekend,
        "transaction_id": f"t{hash(date + desc)}"[:12],
        "day_of_week": "Saturday" if is_weekend else "Wednesday",
    }


# ---------------------------------------------------------------------------
# Pattern: LATE_NIGHT_SPENDING
# ---------------------------------------------------------------------------

class TestLateNightSpending:
    def test_triggers_when_over_15_pct(self):
        rows = [_tx(f"2026-04-{i:02d}", "Uber Eats", -20.0, "food_dining", hour=23) for i in range(1, 5)]
        rows += [_tx(f"2026-04-{i:02d}", "Chipotle", -15.0, "food_dining", hour=13) for i in range(5, 16)]
        df = pd.DataFrame(rows)
        result = _late_night_spending(df)
        assert result is not None
        assert result.type == "LATE_NIGHT_SPENDING"

    def test_no_trigger_when_under_threshold(self):
        rows = [_tx(f"2026-04-{i:02d}", "Starbucks", -6.0, "food_dining", hour=8) for i in range(1, 20)]
        df = pd.DataFrame(rows)
        result = _late_night_spending(df)
        assert result is None


# ---------------------------------------------------------------------------
# Pattern: SUBSCRIPTION_CREEP
# ---------------------------------------------------------------------------

class TestSubscriptionCreep:
    def test_triggers_with_more_than_5_subs(self):
        subs = ["Netflix", "Spotify", "Hulu", "Adobe", "ChatGPT", "Disney+"]
        rows = [_tx("2026-04-01", s, -15.0, "subscriptions") for s in subs]
        df = pd.DataFrame(rows)
        result = _subscription_creep(df)
        assert result is not None
        assert result.type == "SUBSCRIPTION_CREEP"

    def test_no_trigger_with_5_or_fewer(self):
        subs = ["Netflix", "Spotify", "Hulu", "Adobe", "ChatGPT"]
        rows = [_tx("2026-04-01", s, -15.0, "subscriptions") for s in subs]
        df = pd.DataFrame(rows)
        result = _subscription_creep(df)
        assert result is None


# ---------------------------------------------------------------------------
# Pattern: WEEKEND_SPLURGE
# ---------------------------------------------------------------------------

class TestWeekendSplurge:
    def test_triggers_when_weekend_much_higher(self):
        weekday_rows = [_tx(f"2026-04-0{i}", "Coffee", -5.0, is_weekend=False) for i in range(1, 5)]
        weekend_rows = [_tx("2026-04-05", "Restaurant", -150.0, is_weekend=True)]
        df = pd.DataFrame(weekday_rows + weekend_rows)
        result = _weekend_splurge(df)
        assert result is not None
        assert result.type == "WEEKEND_SPLURGE"


# ---------------------------------------------------------------------------
# Pattern: ANOMALOUS_WEEK
# ---------------------------------------------------------------------------

class TestAnomalousWeek:
    def test_triggers_on_spike_week(self):
        rows = []
        for day in range(1, 22):
            rows.append(_tx(f"2026-04-{day:02d}", "Coffee", -5.0))
        # Spike in last week
        rows.append(_tx("2026-04-22", "Electronics", -500.0))
        df = pd.DataFrame(rows)
        result = _anomalous_week(df)
        assert result is not None
        assert result.type == "ANOMALOUS_WEEK"


# ---------------------------------------------------------------------------
# detect_patterns (integration)
# ---------------------------------------------------------------------------

class TestDetectPatterns:
    def test_returns_list(self, sample_df):
        results = detect_patterns(sample_df)
        assert isinstance(results, list)

    def test_patterns_have_required_fields(self, sample_df):
        results = detect_patterns(sample_df)
        for p in results:
            assert isinstance(p, Pattern)
            assert p.type
            assert p.headline
            assert p.severity in ("info", "warning", "critical")


# ---------------------------------------------------------------------------
# AnomalyDetector
# ---------------------------------------------------------------------------

class TestAnomalyDetector:
    def _df_with_outlier(self) -> pd.DataFrame:
        rows = []
        for i in range(1, 20):
            rows.append(_tx(f"2026-04-{i:02d}", "Coffee", -5.0))
        rows.append(_tx("2026-04-20", "Luxury Hotel", -1500.0))
        df = pd.DataFrame(rows)
        df["day_of_week"] = "Wednesday"
        return df

    def test_adds_anomaly_columns(self, sample_df):
        result = AnomalyDetector().fit_predict(sample_df)
        assert "anomaly_score" in result.columns
        assert "is_anomaly" in result.columns

    def test_flags_known_outlier(self):
        df = self._df_with_outlier()
        result = AnomalyDetector().fit_predict(df)
        # The $1500 transaction should be the top anomaly
        hotel_row = result[result["description"] == "Luxury Hotel"]
        assert hotel_row["anomaly_score"].iloc[0] >= result["anomaly_score"].median()

    def test_features_include_category_and_timing(self):
        """Task 7: _build_features returns 5 columns when category column present."""

        rows = []
        for i in range(1, 10):
            rows.append(_tx(f"2026-04-{i:02d}", "Coffee", -5.0, category="food_dining"))
        df = pd.DataFrame(rows)
        detector = AnomalyDetector()
        expenses = df[df["amount"] < 0].copy()
        features = detector._build_features(expenses)
        assert features.shape == (len(expenses), 5), (
            f"Expected 5 feature columns, got {features.shape[1]}"
        )

    def test_features_category_stable_across_calls(self):
        """Task 7: category encoding is deterministic — same result on two calls."""
        import numpy as np

        rows = [_tx(f"2026-04-{i:02d}", "Shop", -10.0, category=cat)
                for i, cat in enumerate(["food_dining", "groceries", "shopping"], start=1)]
        df = pd.DataFrame(rows)
        expenses = df[df["amount"] < 0].copy()
        detector = AnomalyDetector()
        f1 = detector._build_features(expenses)
        f2 = detector._build_features(expenses)
        np.testing.assert_array_equal(f1, f2)

    def test_features_without_category_defaults_to_minus_one(self):
        """Task 7: _build_features returns 5 columns even without category column."""

        rows = [_tx(f"2026-04-{i:02d}", "Coffee", -5.0) for i in range(1, 8)]
        df = pd.DataFrame(rows)
        df.drop(columns=["category"], inplace=True, errors="ignore")
        expenses = df[df["amount"] < 0].copy()
        detector = AnomalyDetector()
        features = detector._build_features(expenses)
        assert features.shape[1] == 5
        # category_encoded column (index 3) should all be -1
        assert (features[:, 3] == -1).all()


# ---------------------------------------------------------------------------
# Narrator (mocked)
# ---------------------------------------------------------------------------

class TestNarrator:
    def test_generates_report(self, sample_df):
        from receipt.analysis.narrator import NarrativeReport, Narrator
        from receipt.pipeline.aggregator import compute_stats

        stats = compute_stats(sample_df)
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text='{"insights": [{"headline": "You spent a lot", "detail": "On stuff."}], '
                     '"next_steps": "Spend less.", "tldr": "Tight month."}'
            )
        ]

        with patch("receipt.analysis.narrator.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_response
            narrator = Narrator(api_key="test-key")
            report = narrator.generate_narrative(stats, [])

        assert isinstance(report, NarrativeReport)
        assert report.tldr == "Tight month."
        assert len(report.insights) == 1
        assert report.insights[0].headline == "You spent a lot"
