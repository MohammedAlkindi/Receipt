"""Spending aggregation and statistical summaries.

NOTE: by_category intentionally omits raw transaction lists.
Transaction-level detail grows O(N) with dataset size and is never
used by the narrator, API response, or storage layer.
Use get_transactions() from ReceiptStore for transaction-level queries.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def compute_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute a comprehensive stats dict from a categorized transaction DataFrame."""
    expenses = df[df["amount"] < 0].copy()
    income = df[df["amount"] > 0].copy()

    total_spent = float(expenses["amount"].sum())  # negative
    total_income = float(income["amount"].sum())
    net = total_spent + total_income

    # --- By category ---
    by_category: dict[str, Any] = {}
    if "category" in df.columns:
        for cat, group in expenses.groupby("category"):
            amounts = group["amount"].abs()
            by_category[cat] = {
                "total": float(amounts.sum()),
                "count": int(len(group)),
                "avg": float(amounts.mean()),
            }

    # --- By week ---
    df_copy = df.copy()
    df_copy["week"] = df_copy["date"].dt.tz_localize(None).dt.to_period("W").astype(str)
    by_week: dict[str, float] = {}
    for week, group in df_copy[df_copy["amount"] < 0].groupby("week"):
        by_week[str(week)] = float(group["amount"].abs().sum())

    # --- By merchant ---
    merchant_totals = (
        expenses.groupby("description")["amount"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "total", "count": "count"})
    )
    merchant_totals["total"] = merchant_totals["total"].abs()
    merchant_totals = merchant_totals.sort_values("total", ascending=False)
    by_merchant = merchant_totals.head(20).to_dict(orient="index")

    # --- Subscription total ---
    subscription_total = 0.0
    if "category" in df.columns:
        sub_mask = df["category"] == "subscriptions"
        subscription_total = float(df[sub_mask & (df["amount"] < 0)]["amount"].abs().sum())

    # --- Largest single transaction ---
    largest = None
    if not expenses.empty:
        idx = expenses["amount"].idxmin()
        row = expenses.loc[idx]
        largest = {
            "description": str(row["description"]),
            "amount": float(row["amount"]),
            "date": str(row["date"]),
        }

    # --- Most frequent merchant ---
    most_frequent = None
    if not expenses.empty:
        freq = expenses["description"].value_counts()
        most_frequent = {"merchant": str(freq.index[0]), "count": int(freq.iloc[0])}

    return {
        "total_spent": total_spent,
        "total_income": total_income,
        "net": net,
        "by_category": by_category,
        "by_week": by_week,
        "by_merchant": by_merchant,
        "subscription_total": subscription_total,
        "largest_single_transaction": largest,
        "most_frequent_merchant": most_frequent,
        "transaction_count": len(df),
        "expense_count": len(expenses),
        "income_count": len(income),
    }
