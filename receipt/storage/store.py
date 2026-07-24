"""SQLite persistence engine."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from receipt.storage.models import AnalysisRun, Base, Merchant, Transaction

logger = logging.getLogger(__name__)

_NARRATIVE_SCHEMA_VERSION = 1


class ReceiptStore:
    """Persist transactions, analysis runs, and merchant summaries to SQLite."""

    def __init__(self, db_path: Path | None = None):
        if db_path is None:
            db_path = Path.home() / ".receipt" / "receipt.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(f"sqlite:///{db_path}", echo=False)
        # Schema is managed by Alembic — run `alembic upgrade head` after install
        # and after each upgrade to apply migrations.
        Base.metadata.create_all(self._engine)

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    def save_transactions(self, df: pd.DataFrame, run_id: str) -> int:
        """Upsert transactions from *df* and associate them with *run_id*.

        Returns number of new rows written (skips duplicates by transaction_id).
        Bulk-fetches existing IDs in one query, then batch-inserts new rows.
        """
        if df.empty:
            return 0

        all_ids = df["transaction_id"].astype(str).tolist()

        with Session(self._engine) as session:
            existing_ids: set[str] = set(
                session.execute(
                    select(Transaction.transaction_id).where(
                        Transaction.transaction_id.in_(all_ids)
                    )
                ).scalars().all()
            )

            new_df = df[~df["transaction_id"].astype(str).isin(existing_ids)]
            if new_df.empty:
                return 0

            cols = set(new_df.columns)
            mappings = [
                {
                    "transaction_id": str(row["transaction_id"]),
                    "date": _to_utc_datetime(row["date"]),
                    "description": str(row.get("description", "")),
                    "raw_description": str(row.get("raw_description", "")),
                    "amount": float(row["amount"]),
                    "source": str(row.get("source", "unknown")),
                    "category": str(row["category"]) if "category" in cols and pd.notna(row["category"]) else None,
                    "category_confidence": float(row["category_confidence"]) if "category_confidence" in cols else None,
                    "cluster_id": int(row["cluster_id"]) if "cluster_id" in cols and pd.notna(row["cluster_id"]) else None,
                    "anomaly_score": float(row["anomaly_score"]) if "anomaly_score" in cols else None,
                    "is_anomaly": bool(row["is_anomaly"]) if "is_anomaly" in cols else False,
                    "is_weekend": bool(row["is_weekend"]) if "is_weekend" in cols else None,
                    "day_of_week": str(row["day_of_week"]) if "day_of_week" in cols else None,
                    "run_id": run_id,
                }
                for _, row in new_df.iterrows()
            ]

            session.execute(sqlite_insert(Transaction), mappings)
            session.commit()

        return len(mappings)

    def get_transactions(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Return transactions within the given date range as a DataFrame."""
        with Session(self._engine) as session:
            stmt = select(Transaction)
            if start_date:
                stmt = stmt.where(Transaction.date >= start_date)
            if end_date:
                stmt = stmt.where(Transaction.date <= end_date)
            rows = session.execute(stmt).scalars().all()

        if not rows:
            return pd.DataFrame()

        records = [
            {
                "transaction_id": r.transaction_id,
                "date": r.date,
                "description": r.description,
                "raw_description": r.raw_description,
                "amount": r.amount,
                "source": r.source,
                "category": r.category,
                "category_confidence": r.category_confidence,
                "anomaly_score": r.anomaly_score,
                "is_anomaly": r.is_anomaly,
            }
            for r in rows
        ]
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Analysis runs
    # ------------------------------------------------------------------

    def save_analysis(
        self,
        period_start: datetime,
        period_end: datetime,
        transaction_count: int,
        source_file: str | None = None,
        narrative: dict[str, Any] | None = None,
    ) -> str:
        """Persist an AnalysisRun record; return its run_id."""
        run_id = uuid.uuid4().hex[:12]
        with Session(self._engine) as session:
            run = AnalysisRun(
                run_id=run_id,
                created_at=datetime.now(timezone.utc),
                period_start=period_start,
                period_end=period_end,
                source_file=source_file,
                transaction_count=transaction_count,
                narrative_json=json.dumps(narrative) if narrative else None,
            )
            session.add(run)
            session.commit()
        return run_id

    def get_previous_period(self, current_start: datetime) -> pd.DataFrame | None:
        """Return transactions from the period immediately before *current_start*."""
        with Session(self._engine) as session:
            latest_run = session.execute(
                select(AnalysisRun)
                .where(AnalysisRun.period_end < current_start)
                .order_by(AnalysisRun.period_end.desc())
                .limit(1)
            ).scalar_one_or_none()
            if not latest_run:
                return None
            return self.get_transactions(latest_run.period_start, latest_run.period_end)

    def get_analysis_history(self) -> list[dict[str, Any]]:
        """Return all analysis runs, newest first, without transaction details."""
        with Session(self._engine) as session:
            runs = session.execute(
                select(AnalysisRun).order_by(AnalysisRun.created_at.desc())
            ).scalars().all()
        result = []
        for r in runs:
            narrative = _deserialize_narrative(json.loads(r.narrative_json)) if r.narrative_json else None
            result.append(
                {
                    "run_id": r.run_id,
                    "created_at": r.created_at.isoformat(),
                    "period_start": r.period_start.isoformat(),
                    "period_end": r.period_end.isoformat(),
                    "source_file": r.source_file,
                    "transaction_count": r.transaction_count,
                    "tldr": narrative.get("tldr") if narrative else None,
                }
            )
        return result

    def get_analysis_run(self, run_id: str) -> dict[str, Any] | None:
        with Session(self._engine) as session:
            run = session.execute(
                select(AnalysisRun).where(AnalysisRun.run_id == run_id)
            ).scalar_one_or_none()
            if not run:
                return None
            narrative = _deserialize_narrative(json.loads(run.narrative_json)) if run.narrative_json else None
            return {
                "run_id": run.run_id,
                "created_at": run.created_at.isoformat(),
                "period_start": run.period_start.isoformat(),
                "period_end": run.period_end.isoformat(),
                "source_file": run.source_file,
                "transaction_count": run.transaction_count,
                "narrative": narrative,
            }

    # ------------------------------------------------------------------
    # Merchants
    # ------------------------------------------------------------------

    def upsert_merchants(self, df: pd.DataFrame) -> None:
        """Update merchant summary table from a categorized transaction DataFrame.

        Uses INSERT ... ON CONFLICT DO UPDATE (upsert) to avoid race conditions
        on concurrent writes to the same normalized_name.
        """
        expenses = df[df["amount"] < 0].copy()
        if expenses.empty:
            return
        grouped = (
            expenses.groupby("description")
            .agg(
                total=("amount", lambda x: x.abs().sum()),
                count=("amount", "count"),
                first_seen=("date", "min"),
                last_seen=("date", "max"),
            )
            .reset_index()
        )
        if "category" in expenses.columns:
            cat_map = expenses.groupby("description")["category"].first().to_dict()
            grouped["category"] = grouped["description"].map(cat_map)

        mappings = []
        for _, row in grouped.iterrows():
            norm = str(row["description"]).lower().strip()
            has_cat = "category" in grouped.columns
            cat_val = (
                str(row["category"]) if has_cat and pd.notna(row["category"]) else None
            )
            mappings.append(
                {
                    "name": str(row["description"]),
                    "normalized_name": norm,
                    "category": cat_val,
                    "first_seen": _to_utc_datetime(row["first_seen"]),
                    "last_seen": _to_utc_datetime(row["last_seen"]),
                    "total_spent": float(row["total"]),
                    "transaction_count": int(row["count"]),
                }
            )

        if not mappings:
            return

        with Session(self._engine) as session:
            stmt = sqlite_insert(Merchant).values(mappings)
            stmt = stmt.on_conflict_do_update(
                index_elements=["normalized_name"],
                set_={
                    "total_spent": Merchant.total_spent + stmt.excluded.total_spent,
                    "transaction_count": Merchant.transaction_count + stmt.excluded.transaction_count,
                    "last_seen": stmt.excluded.last_seen,
                },
            )
            session.execute(stmt)
            session.commit()

    def get_merchants(self, limit: int = 50) -> list[dict[str, Any]]:
        with Session(self._engine) as session:
            merchants = session.execute(
                select(Merchant).order_by(Merchant.total_spent.desc()).limit(limit)
            ).scalars().all()
        return [
            {
                "name": m.normalized_name,
                "category": m.category,
                "total_spent": m.total_spent,
                "transaction_count": m.transaction_count,
                "first_seen": m.first_seen.isoformat() if m.first_seen else None,
                "last_seen": m.last_seen.isoformat() if m.last_seen else None,
            }
            for m in merchants
        ]

    def db_stats(self) -> dict[str, int]:
        from sqlalchemy import func as sqlfunc

        with Session(self._engine) as session:
            return {
                "transactions": session.execute(
                    select(sqlfunc.count()).select_from(Transaction)
                ).scalar() or 0,
                "analysis_runs": session.execute(
                    select(sqlfunc.count()).select_from(AnalysisRun)
                ).scalar() or 0,
                "merchants": session.execute(
                    select(sqlfunc.count()).select_from(Merchant)
                ).scalar() or 0,
            }


def _deserialize_narrative(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate narrative dicts across schema versions.

    Returns the best available data — never raises.
    """
    try:
        version = data.get("schema_version", 0)
        if version == 0:
            # Legacy: ensure insights is a list of {headline, detail} dicts
            insights = data.get("insights", [])
            if not isinstance(insights, list):
                insights = []
            data = {
                "schema_version": 0,
                "tldr": data.get("tldr", ""),
                "insights": insights,
                "next_steps": data.get("next_steps", ""),
                "generated_at": data.get("generated_at", ""),
            }
        elif version == 1:
            pass  # current — return as-is
        else:
            logger.warning(
                "Narrative schema_version %d is newer than supported (%d); returning as-is.",
                version,
                _NARRATIVE_SCHEMA_VERSION,
            )
        return data
    except Exception:
        return {}


def _to_utc_datetime(val: Any) -> datetime:
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=timezone.utc)
        return val
    if hasattr(val, "to_pydatetime"):
        dt = val.to_pydatetime()
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    return datetime.fromisoformat(str(val)).replace(tzinfo=timezone.utc)
