"""Bank of America CSV parser."""

from __future__ import annotations

from io import IOBase
from pathlib import Path
from typing import Union

import pandas as pd

from receipt.ingestion.base import MAX_ROWS, ParseError, TransactionParser

_BOFA_REQUIRED = {"Date", "Description", "Amount"}
_BOFA_OPTIONAL = {"Running Bal."}


class BofAParser(TransactionParser):
    """Parse Bank of America checking/savings CSV exports.

    BofA format:
    - Date column: MM/DD/YYYY
    - Amount column: signed float (negative = debit)
    - Optional 'Running Bal.' column (ignored for analysis)
    """

    @classmethod
    def detect(cls, df: pd.DataFrame) -> bool:
        cols = {c.strip() for c in df.columns}
        return _BOFA_REQUIRED.issubset(cols) and "Running Bal." in cols

    def parse(self, source: Union[str, Path, IOBase]) -> pd.DataFrame:
        try:
            if isinstance(source, (str, Path)):
                raw = pd.read_csv(source, encoding="utf-8-sig", thousands=",", nrows=MAX_ROWS + 1)
            else:
                source.seek(0)
                raw = pd.read_csv(source, encoding="utf-8-sig", thousands=",", nrows=MAX_ROWS + 1)
        except Exception as exc:
            raise ParseError(f"BofA parser failed to read file: {exc}") from exc

        raw.columns = [c.strip() for c in raw.columns]
        missing = _BOFA_REQUIRED - set(raw.columns)
        if missing:
            raise ParseError(f"BofA CSV missing required columns: {missing}")

        result = pd.DataFrame()
        # BofA dates are MM/DD/YYYY
        result["date"] = pd.to_datetime(raw["Date"], format="%m/%d/%Y", errors="coerce")
        result["description"] = raw["Description"].fillna("").astype(str)
        result["raw_description"] = result["description"]
        result["amount"] = pd.to_numeric(
            raw["Amount"].astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce",
        )

        return self._finalise(result, source_name="bofa")
