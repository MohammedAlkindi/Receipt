"""Plaid CSV export parser."""

from __future__ import annotations

from io import IOBase
from pathlib import Path
from typing import Union

import pandas as pd

from receipt.ingestion.base import MAX_ROWS, ParseError, TransactionParser

_PLAID_REQUIRED = {"date", "name", "amount"}
_PLAID_OPTIONAL = {"category", "account_id", "pending"}


class PlaidParser(TransactionParser):
    """Parse Plaid CSV exports.

    Plaid convention: positive amount = expense, negative = income.
    We invert the sign to match the standard schema (negative = expense).

    The 'pending' column is used to filter unposted transactions.
    """

    @classmethod
    def detect(cls, df: pd.DataFrame) -> bool:
        cols = {c.strip().lower() for c in df.columns}
        return _PLAID_REQUIRED.issubset(cols)

    def parse(self, source: Union[str, Path, IOBase]) -> pd.DataFrame:
        try:
            if isinstance(source, (str, Path)):
                raw = pd.read_csv(source, encoding="utf-8-sig", nrows=MAX_ROWS + 1)
            else:
                source.seek(0)
                raw = pd.read_csv(source, encoding="utf-8-sig", nrows=MAX_ROWS + 1)
        except Exception as exc:
            raise ParseError(f"Plaid parser failed to read file: {exc}") from exc

        raw.columns = [c.strip().lower() for c in raw.columns]
        missing = _PLAID_REQUIRED - set(raw.columns)
        if missing:
            raise ParseError(f"Plaid CSV missing required columns: {missing}")

        # Filter out pending transactions
        if "pending" in raw.columns:
            raw = raw[raw["pending"].astype(str).str.lower() != "true"].copy()

        result = pd.DataFrame()
        result["date"] = raw["date"]
        result["description"] = raw["name"].fillna("").astype(str)
        result["raw_description"] = result["description"]

        # Invert: Plaid positive = expense, standard schema negative = expense
        result["amount"] = pd.to_numeric(raw["amount"], errors="coerce") * -1

        return self._finalise(result, source_name="plaid")
