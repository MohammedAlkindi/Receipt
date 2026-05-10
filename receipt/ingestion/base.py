"""Abstract base class for all transaction parsers."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from io import IOBase
from pathlib import Path
from typing import Union

import pandas as pd


class ParseError(Exception):
    """Raised when a file cannot be parsed with an actionable message."""


class TransactionParser(ABC):
    """Base class every bank parser must implement.

    The standard output DataFrame always has these columns:
        date             datetime64[ns, UTC]
        description      str   — normalized merchant name
        amount           float — negative = expense, positive = income
        raw_description  str   — original text before normalization
        source           str   — bank identifier string
        transaction_id   str   — deterministic hash
    """

    STANDARD_SCHEMA: dict[str, str] = {
        "date": "datetime64[ns, UTC]",
        "description": "object",
        "amount": "float64",
        "raw_description": "object",
        "source": "object",
        "transaction_id": "object",
    }

    @abstractmethod
    def parse(self, source: Union[str, Path, IOBase]) -> pd.DataFrame:
        """Parse *source* and return a standardised DataFrame."""

    def get_schema(self) -> dict:
        return self.STANDARD_SCHEMA.copy()

    # ------------------------------------------------------------------
    # Helpers shared by all subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def make_transaction_id(date: str, description: str, amount: float) -> str:
        """Deterministic SHA-256-derived ID from the three key fields."""
        payload = f"{date}|{description}|{amount:.4f}".encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    @staticmethod
    def _coerce_to_utc(series: pd.Series) -> pd.Series:
        """Parse a date Series and ensure UTC timezone."""
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.dt.tz is None:
            parsed = parsed.dt.tz_localize("UTC")
        else:
            parsed = parsed.dt.tz_convert("UTC")
        return parsed

    def _finalise(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Apply common post-processing: source, IDs, schema enforcement."""
        df = df.copy()
        df["source"] = source_name
        df["raw_description"] = df.get("raw_description", df["description"])
        df["transaction_id"] = df.apply(
            lambda r: self.make_transaction_id(
                str(r["date"]), str(r["raw_description"]), float(r["amount"])
            ),
            axis=1,
        )
        df["date"] = self._coerce_to_utc(df["date"])
        df["amount"] = df["amount"].astype(float)
        df["description"] = df["description"].astype(str)
        df["raw_description"] = df["raw_description"].astype(str)
        return df[list(self.STANDARD_SCHEMA.keys())]

    def validate(self, df: pd.DataFrame) -> bool:
        required = set(self.STANDARD_SCHEMA.keys())
        return required.issubset(set(df.columns)) and len(df) > 0
