"""Longitudinal behavioral drift detection between two periods."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Structured diff between two spending periods."""

    increased: dict[str, dict[str, Any]] = field(default_factory=dict)
    decreased: dict[str, dict[str, Any]] = field(default_factory=dict)
    new_merchants: list[str] = field(default_factory=list)
    dropped_merchants: list[str] = field(default_factory=list)
    velocity_trend: str = "stable"  # "accelerating" | "decelerating" | "stable"
    subscription_drift: dict[str, list[str]] = field(default_factory=dict)
    narrative_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "increased": self.increased,
            "decreased": self.decreased,
            "new_merchants": self.new_merchants,
            "dropped_merchants": self.dropped_merchants,
            "velocity_trend": self.velocity_trend,
            "subscription_drift": self.subscription_drift,
            "narrative_hints": self.narrative_hints,
        }


class DriftDetector:
    """Compare two transaction DataFrames and surface behavioral changes."""

    DRIFT_THRESHOLD = 0.20  # kept for backwards-compatible class-level access

    def __init__(self, drift_threshold: float = 0.20) -> None:
        if not 0.0 <= drift_threshold <= 1.0:
            raise ValueError(
                f"drift_threshold must be between 0.0 and 1.0, got {drift_threshold}"
            )
        self.drift_threshold = drift_threshold

    def compare_periods(
        self, df_current: pd.DataFrame, df_previous: pd.DataFrame
    ) -> DriftReport:
        report = DriftReport()

        skip_category_drift = False
        if "category" not in df_current.columns or "category" not in df_previous.columns:
            logger.warning(
                "DriftDetector.compare_periods: 'category' column missing. "
                "Call SemanticCategorizer.categorize() before drift detection. "
                "Category drift analysis will be skipped."
            )
            skip_category_drift = True

        # --- Category drift ---
        if not skip_category_drift:
            cat_curr = self._category_totals(df_current)
            cat_prev = self._category_totals(df_previous)
            all_cats = set(cat_curr) | set(cat_prev)
        else:
            cat_curr = cat_prev = {}
            all_cats = set()

        for cat in all_cats:
            curr_val = cat_curr.get(cat, 0.0)
            prev_val = cat_prev.get(cat, 0.0)
            if prev_val == 0:
                continue
            change = (curr_val - prev_val) / abs(prev_val)
            detail = {
                "current": round(curr_val, 2),
                "previous": round(prev_val, 2),
                "change_pct": round(change * 100, 1),
            }
            if change > self.drift_threshold:
                report.increased[cat] = detail
                report.narrative_hints.append(
                    f"{cat} spending rose {detail['change_pct']}% "
                    f"(${detail['previous']} → ${detail['current']})"
                )
            elif change < -self.drift_threshold:
                report.decreased[cat] = detail
                report.narrative_hints.append(
                    f"{cat} spending fell {abs(detail['change_pct'])}% "
                    f"(${detail['previous']} → ${detail['current']})"
                )

        # --- Merchant drift ---
        curr_merchants = set(df_current["description"].unique())
        prev_merchants = set(df_previous["description"].unique())
        report.new_merchants = sorted(curr_merchants - prev_merchants)
        report.dropped_merchants = sorted(prev_merchants - curr_merchants)

        if report.new_merchants:
            report.narrative_hints.append(
                f"New merchants this period: {', '.join(report.new_merchants[:5])}"
            )

        # --- Spending velocity: compare first half vs second half of current ---
        report.velocity_trend = self._spending_velocity(df_current)

        # --- Subscription drift ---
        if "category" in df_current.columns and "category" in df_previous.columns:
            curr_subs = set(
                df_current[df_current["category"] == "subscriptions"]["description"].unique()
            )
            prev_subs = set(
                df_previous[df_previous["category"] == "subscriptions"]["description"].unique()
            )
            new_subs = sorted(curr_subs - prev_subs)
            cancelled_subs = sorted(prev_subs - curr_subs)
            report.subscription_drift = {"new": new_subs, "cancelled": cancelled_subs}

            if new_subs:
                report.narrative_hints.append(f"New subscriptions: {', '.join(new_subs)}")
            if cancelled_subs:
                report.narrative_hints.append(
                    f"Cancelled subscriptions: {', '.join(cancelled_subs)}"
                )

        return report

    @staticmethod
    def _category_totals(df: pd.DataFrame) -> dict[str, float]:
        if "category" not in df.columns:
            return {}
        expenses = df[df["amount"] < 0]
        totals = expenses.groupby("category")["amount"].sum().abs()
        return totals.to_dict()

    @staticmethod
    def _spending_velocity(df: pd.DataFrame) -> str:
        """Compare spending in first vs second half of the period."""
        if df.empty:
            return "stable"
        expenses = df[df["amount"] < 0].copy()
        if expenses.empty:
            return "stable"
        mid = expenses["date"].min() + (expenses["date"].max() - expenses["date"].min()) / 2
        first_half = expenses[expenses["date"] <= mid]["amount"].abs().sum()
        second_half = expenses[expenses["date"] > mid]["amount"].abs().sum()
        if second_half == 0:
            return "stable"
        ratio = float(second_half / first_half) if first_half > 0 else 1.0
        if ratio > 1.25:
            return "accelerating"
        if ratio < 0.75:
            return "decelerating"
        return "stable"
