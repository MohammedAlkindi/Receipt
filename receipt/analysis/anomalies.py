"""Anomaly detection using Isolation Forest."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Flag statistically unusual transactions using Isolation Forest.

    Features used: amount, day_of_week (encoded), hour.
    Falls back to z-score if sklearn is unavailable.
    """

    def __init__(self, contamination: float = 0.05, top_n: int = 3):
        self.contamination = contamination
        self.top_n = top_n

    def fit_predict(
        self,
        df: pd.DataFrame,
        audit_log: "Any | None" = None,
    ) -> pd.DataFrame:
        """Return df with added columns: anomaly_score, is_anomaly, anomaly_reason."""
        from receipt.pipeline.audit import AuditLogger

        with AuditLogger(audit_log, "fit_predict", len(df)) as al:
            result = self._fit_predict_impl(df)
            al.output_rows = len(result)
            al.metadata["anomalies_found"] = int(result["is_anomaly"].sum()) if "is_anomaly" in result.columns else 0
        return result

    def _fit_predict_impl(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        expenses = df[df["amount"] < 0].copy()
        if len(expenses) < 5:
            df["anomaly_score"] = 0.0
            df["is_anomaly"] = False
            df["anomaly_reason"] = ""
            return df

        features = self._build_features(expenses)
        scores = self._compute_scores(features)

        df["anomaly_score"] = 0.0
        df.loc[expenses.index, "anomaly_score"] = scores

        threshold = float(np.percentile(scores, (1 - self.contamination) * 100))
        df["is_anomaly"] = df["anomaly_score"] >= threshold

        # Label top anomalies with reasons
        df["anomaly_reason"] = ""
        top_idx = expenses.loc[
            expenses.index[np.argsort(scores)[-self.top_n :]]
        ].index
        for idx in top_idx:
            row = df.loc[idx]
            reason = self._explain(row, expenses)
            df.at[idx, "anomaly_reason"] = reason
            df.at[idx, "is_anomaly"] = True

        return df

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        features = pd.DataFrame(index=df.index)
        features["amount_abs"] = df["amount"].abs()
        if "day_of_week" in df.columns:
            dow_map = {
                "Monday": 0, "Tuesday": 1, "Wednesday": 2,
                "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6,
            }
            features["dow"] = df["day_of_week"].map(dow_map).fillna(0)
        else:
            features["dow"] = 0
        features["hour"] = df.get("hour", pd.Series(0, index=df.index))

        # Category encoded: stable alphabetical integer encoding; "other" → -1
        if "category" in df.columns:
            unique_cats = sorted(c for c in df["category"].dropna().unique() if c != "other")
            cat_index = {cat: i for i, cat in enumerate(unique_cats)}
            features["category_encoded"] = (
                df["category"].map(lambda c: cat_index.get(c, -1) if c != "other" else -1)
            )
        else:
            features["category_encoded"] = -1

        # Days since previous transaction (sorted by date); first row = 0
        df_sorted = df.sort_values("date") if "date" in df.columns else df
        if "date" in df_sorted.columns:
            prev_date = df_sorted["date"].shift(1)
            gap = (df_sorted["date"] - prev_date).dt.total_seconds().div(86400).fillna(0)
            features["days_since_prev"] = gap.reindex(features.index).fillna(0)
        else:
            features["days_since_prev"] = 0

        return features.fillna(0).values

    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        try:
            from sklearn.ensemble import IsolationForest

            model = IsolationForest(
                contamination=self.contamination, random_state=42, n_estimators=100
            )
            model.fit(X)
            # IsolationForest returns -1 for anomalies; convert to [0, 1] score
            decision = model.decision_function(X)
            scores = -decision  # higher = more anomalous
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
            return scores
        except Exception as exc:
            logger.warning("IsolationForest failed (%s); using z-score fallback.", exc)
            amounts = X[:, 0]
            mean, std = amounts.mean(), amounts.std()
            return np.abs((amounts - mean) / (std + 1e-9))

    @staticmethod
    def _explain(row: "pd.Series[Any]", all_expenses: pd.DataFrame) -> str:
        amount = abs(float(row["amount"]))
        merchant = str(row.get("description", "Unknown"))
        median_amount = float(all_expenses["amount"].abs().median())
        if amount > median_amount * 5:
            return f"${amount:.2f} at {merchant} is {amount / median_amount:.1f}x the median transaction"
        if "is_weekend" in row and row["is_weekend"]:
            return f"Unusually large weekend purchase: ${amount:.2f} at {merchant}"
        return f"${amount:.2f} at {merchant} is an outlier for this period"
