"""Chase bank CSV parser."""

from __future__ import annotations

from io import IOBase
from pathlib import Path
from typing import Union

import pandas as pd

from receipt.ingestion.base import MAX_ROWS, ParseError, TransactionParser

_CHASE_REQUIRED = {"Transaction Date", "Description", "Amount"}
_CHASE_OPTIONAL = {"Post Date", "Category", "Type", "Memo"}


class ChaseParser(TransactionParser):
    """Parse Chase credit-card and checking CSV exports.

    Chase uses a single signed Amount column (negative = debit/expense).
    The 'Type' column distinguishes Sale / Payment / Fee.
    """

    @classmethod
    def detect(cls, df: pd.DataFrame) -> bool:
        cols = {c.strip() for c in df.columns}
        return _CHASE_REQUIRED.issubset(cols)

    def parse(self, source: Union[str, Path, IOBase]) -> pd.DataFrame:
        try:
            if isinstance(source, (str, Path)):
                raw = pd.read_csv(source, encoding="utf-8-sig", thousands=",", nrows=MAX_ROWS + 1)
            else:
                source.seek(0)
                raw = pd.read_csv(source, encoding="utf-8-sig", thousands=",", nrows=MAX_ROWS + 1)
        except Exception as exc:
            raise ParseError(f"Chase parser failed to read file: {exc}") from exc

        raw.columns = [c.strip() for c in raw.columns]
        missing = _CHASE_REQUIRED - set(raw.columns)
        if missing:
            raise ParseError(f"Chase CSV missing required columns: {missing}")

        result = pd.DataFrame()
        result["date"] = raw["Transaction Date"]
        result["description"] = raw["Description"].fillna("").astype(str)
        result["raw_description"] = result["description"]
        result["amount"] = pd.to_numeric(
            raw["Amount"].astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce",
        )

        return self._finalise(result, source_name="chase")
