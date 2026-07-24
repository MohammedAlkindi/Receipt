"""Spending pattern detection algorithms."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

Severity = Literal["info", "warning", "critical"]


@dataclass
class Pattern:
    type: str
    headline: str
    data: dict[str, Any]
    severity: Severity


def detect_patterns(df: pd.DataFrame) -> list[Pattern]:
    """Run all pattern detectors and return a flat list of Pattern objects."""
    patterns: list[Pattern] = []
    detectors = [
        _late_night_spending,
        _subscription_creep,
        _weekend_splurge,
        _single_merchant_dominance,
        _anomalous_week,
        _income_irregularity,
        _recurring_forgotten,
    ]
    for detector in detectors:
        result = detector(df)
        if result:
            patterns.extend(result) if isinstance(result, list) else patterns.append(result)
    return patterns


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

def _late_night_spending(df: pd.DataFrame) -> Pattern | None:
    """Flag if more than 15% of food spend happens between 10pm and 4am."""
    if "hour" not in df.columns:
        return None
    # Date-only sources (every bank CSV) parse to hour 0 on all rows, which
    # would count 100% of food spend as "after 10 PM". No real time data,
    # no late-night claim.
    if df["hour"].nunique(dropna=True) <= 1:
        return None
    food_mask = df.get("category", pd.Series()) == "food_dining"
    food = df[food_mask & (df["amount"] < 0)]
    if food.empty:
        return None
    late = food[(food["hour"] >= 22) | (food["hour"] < 4)]
    pct = len(late) / len(food)
    if pct < 0.15:
        return None
    total_late = float(late["amount"].abs().sum())
    return Pattern(
        type="LATE_NIGHT_SPENDING",
        headline=f"{pct:.0%} of food spend happens after 10 PM",
        data={
            "late_night_transaction_count": len(late),
            "total_food_transactions": len(food),
            "late_night_total": total_late,
            "percentage": round(pct * 100, 1),
        },
        severity="info",
    )


def _subscription_creep(df: pd.DataFrame) -> Pattern | None:
    """Flag if more than 5 distinct active subscriptions detected."""
    if "category" not in df.columns:
        return None
    sub_mask = (df["category"] == "subscriptions") & (df["amount"] < 0)
    subs = df[sub_mask]["description"].unique()
    if len(subs) <= 5:
        return None
    total = float(df[sub_mask]["amount"].abs().sum())
    return Pattern(
        type="SUBSCRIPTION_CREEP",
        headline=f"{len(subs)} active subscriptions totalling ${total:.2f}",
        data={
            "subscription_count": len(subs),
            "subscriptions": sorted(subs.tolist()),
            "total_spend": round(total, 2),
        },
        severity="warning",
    )


def _weekend_splurge(df: pd.DataFrame) -> Pattern | None:
    """Compare weekend vs weekday average daily spend."""
    if "is_weekend" not in df.columns:
        return None
    expenses = df[df["amount"] < 0].copy()
    if expenses.empty:
        return None
    weekend = expenses[expenses["is_weekend"]]["amount"].abs()
    weekday = expenses[~expenses["is_weekend"]]["amount"].abs()
    if weekend.empty or weekday.empty:
        return None
    wknd_avg = float(weekend.mean())
    wkdy_avg = float(weekday.mean())
    ratio = wknd_avg / wkdy_avg if wkdy_avg > 0 else 1.0
    if ratio < 1.3:
        return None
    return Pattern(
        type="WEEKEND_SPLURGE",
        headline=f"Weekend spend is {ratio:.1f}x higher than weekdays",
        data={
            "weekend_avg": round(wknd_avg, 2),
            "weekday_avg": round(wkdy_avg, 2),
            "ratio": round(ratio, 2),
        },
        severity="info" if ratio < 2.0 else "warning",
    )


def _single_merchant_dominance(df: pd.DataFrame) -> list[Pattern]:
    """Flag any merchant accounting for >25% of its category's spend."""
    patterns: list[Pattern] = []
    if "category" not in df.columns:
        return patterns
    expenses = df[df["amount"] < 0]
    for cat, group in expenses.groupby("category"):
        cat_total = group["amount"].abs().sum()
        if cat_total == 0:
            continue
        merchant_totals = group.groupby("description")["amount"].sum().abs()
        dominant = merchant_totals[merchant_totals / cat_total > 0.25]
        for merchant, amount in dominant.items():
            pct = amount / cat_total
            patterns.append(
                Pattern(
                    type="SINGLE_MERCHANT_DOMINANCE",
                    headline=f"{merchant} is {pct:.0%} of your {cat} spend",
                    data={
                        "merchant": merchant,
                        "category": cat,
                        "amount": round(float(amount), 2),
                        "category_total": round(float(cat_total), 2),
                        "percentage": round(pct * 100, 1),
                    },
                    severity="info",
                )
            )
    return patterns


def _anomalous_week(df: pd.DataFrame) -> Pattern | None:
    """Flag any week with spend >1.5x the weekly average."""
    expenses = df[df["amount"] < 0].copy()
    if expenses.empty:
        return None
    expenses["_week"] = expenses["date"].dt.tz_localize(None).dt.to_period("W").astype(str)
    weekly = expenses.groupby("_week")["amount"].sum().abs()
    if len(weekly) < 2:
        return None
    avg = float(weekly.mean())
    max_week = weekly.idxmax()
    max_val = float(weekly.max())
    if max_val < avg * 1.5:
        return None
    return Pattern(
        type="ANOMALOUS_WEEK",
        headline=f"Week of {max_week} was {max_val / avg:.1f}x your typical week",
        data={
            "anomalous_week": max_week,
            "week_total": round(max_val, 2),
            "weekly_average": round(avg, 2),
            "multiplier": round(max_val / avg, 2),
        },
        severity="warning",
    )


def _income_irregularity(df: pd.DataFrame) -> Pattern | None:
    """Detect irregular income deposits."""
    if "category" not in df.columns:
        return None
    income_mask = (df["category"] == "income") | (df["amount"] > 0)
    income = df[income_mask & (df["amount"] > 0)].copy()
    # Need at least 3 deposits: 2 deposits yield a single gap whose stdev is
    # NaN, which slipped past the threshold and emitted "±nan days" (and
    # non-JSON-serializable NaN in the pattern data).
    if len(income) < 3:
        return None
    income = income.sort_values("date")
    gaps = income["date"].diff().dt.days.dropna()
    avg_gap = float(gaps.mean())
    std_gap = float(gaps.std())
    # Irregular if std > 5 days (guard against a NaN stdev defensively).
    if math.isnan(std_gap) or std_gap < 5:
        return None
    return Pattern(
        type="INCOME_IRREGULARITY",
        headline=f"Income arrives every {avg_gap:.0f} days on average (±{std_gap:.0f} days)",
        data={
            "income_events": len(income),
            "avg_gap_days": round(avg_gap, 1),
            "std_gap_days": round(std_gap, 1),
            "amounts": income["amount"].tolist(),
        },
        severity="info",
    )


def _recurring_forgotten(df: pd.DataFrame) -> list[Pattern]:
    """Detect same amount + same merchant repeating monthly, outside subscriptions."""
    patterns: list[Pattern] = []
    if "category" not in df.columns:
        return patterns
    expenses = df[(df["amount"] < 0) & (df["category"] != "subscriptions")].copy()
    if expenses.empty:
        return patterns
    expenses["_month"] = expenses["date"].dt.tz_localize(None).dt.to_period("M").astype(str)
    grouped = (
        expenses.groupby(["description", "amount"])["_month"]
        .nunique()
        .reset_index(name="month_count")
    )
    recurring = grouped[grouped["month_count"] >= 2]
    for _, row in recurring.iterrows():
        patterns.append(
            Pattern(
                type="RECURRING_FORGOTTEN",
                headline=f"Possible forgotten recurring: {row['description']} at ${abs(row['amount']):.2f}/month",
                data={
                    "merchant": row["description"],
                    "amount": float(row["amount"]),
                    "months_seen": int(row["month_count"]),
                },
                severity="info",
            )
        )
    return patterns
