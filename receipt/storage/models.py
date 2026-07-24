"""SQLAlchemy ORM models."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    description: Mapped[str] = mapped_column(String(512))
    raw_description: Mapped[str] = mapped_column(String(512))
    amount: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(64))
    category: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    category_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    cluster_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    is_weekend: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    day_of_week: Mapped[str | None] = mapped_column(String(16), nullable=True)
    run_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("analysis_runs.run_id"), nullable=True, index=True)

    run: Mapped[AnalysisRun | None] = relationship("AnalysisRun", back_populates="transactions")

    def __repr__(self) -> str:
        return f"<Transaction {self.transaction_id} {self.description} {self.amount}>"


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    source_file: Mapped[str | None] = mapped_column(String(512), nullable=True)
    transaction_count: Mapped[int] = mapped_column(Integer, default=0)
    narrative_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    transactions: Mapped[list[Transaction]] = relationship(
        "Transaction", back_populates="run"
    )

    def __repr__(self) -> str:
        return f"<AnalysisRun {self.run_id} {self.period_start.date()} – {self.period_end.date()}>"


class Merchant(Base):
    __tablename__ = "merchants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(512), index=True)
    normalized_name: Mapped[str] = mapped_column(String(512), unique=True, index=True)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    first_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    total_spent: Mapped[float] = mapped_column(Float, default=0.0)
    transaction_count: Mapped[int] = mapped_column(Integer, default=0)

    def __repr__(self) -> str:
        return f"<Merchant {self.normalized_name} ${self.total_spent:.2f}>"
