"""Generic CSV parser with automatic column detection."""

from __future__ import annotations

import re
from io import IOBase
from pathlib import Path
from typing import Union

import pandas as pd

from receipt.ingestion.base import ParseError, TransactionParser

# Known column name variants for fuzzy matching
_DATE_VARIANTS = {"date", "transaction date", "posted date", "trans date", "post date"}
_DESC_VARIANTS = {"description", "memo", "details", "payee", "merchant", "name", "transaction"}
_AMOUNT_VARIANTS = {"amount", "transaction amount", "trans amount"}
_DEBIT_VARIANTS = {"debit", "withdrawal", "debit amount", "withdrawals"}
_CREDIT_VARIANTS = {"credit", "deposit", "credit amount", "deposits"}


def _normalise_col(col: str) -> str:
    return re.sub(r"\s+", " ", col.strip().lower())


def _find_col(columns: list[str], variants: set[str]) -> str | None:
    norm = {_normalise_col(c): c for c in columns}
    for variant in variants:
        if variant in norm:
            return norm[variant]
    return None


class GenericCSVParser(TransactionParser):
    """Parse any CSV by fuzzy-matching column names against known variants.

    Handles:
    - Single amount column (negative = debit) or split debit/credit columns
    - UTF-8 BOM, latin-1, and windows-1252 encodings
    - Quoted fields and arbitrary delimiters
    """

    def parse(self, source: Union[str, Path, IOBase]) -> pd.DataFrame:
        df = self._read_raw(source)
        cols = list(df.columns)

        date_col = _find_col(cols, _DATE_VARIANTS)
        desc_col = _find_col(cols, _DESC_VARIANTS)
        amount_col = _find_col(cols, _AMOUNT_VARIANTS)
        debit_col = _find_col(cols, _DEBIT_VARIANTS)
        credit_col = _find_col(cols, _CREDIT_VARIANTS)

        if date_col is None:
            raise ParseError(
                f"Cannot detect a date column. Found columns: {cols}. "
                f"Expected one of: {sorted(_DATE_VARIANTS)}"
            )
        if desc_col is None:
            raise ParseError(
                f"Cannot detect a description column. Found columns: {cols}. "
                f"Expected one of: {sorted(_DESC_VARIANTS)}"
            )
        if amount_col is None and (debit_col is None or credit_col is None):
            raise ParseError(
                f"Cannot detect an amount column. Found columns: {cols}. "
                f"Expected '{sorted(_AMOUNT_VARIANTS)}' or both "
                f"'{sorted(_DEBIT_VARIANTS)}' and '{sorted(_CREDIT_VARIANTS)}'."
            )

        result = pd.DataFrame()
        result["date"] = df[date_col]
        result["description"] = df[desc_col].fillna("").astype(str)
        result["raw_description"] = result["description"]

        if amount_col:
            result["amount"] = self._parse_amount(df[amount_col])
        else:
            debits = self._parse_amount(df[debit_col]).abs() * -1
            credits = self._parse_amount(df[credit_col]).abs()
            result["amount"] = debits.fillna(0) + credits.fillna(0)

        return self._finalise(result, source_name="generic")

    def _read_raw(self, source: Union[str, Path, IOBase]) -> pd.DataFrame:
        encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                if isinstance(source, (str, Path)):
                    return pd.read_csv(source, encoding=enc, thousands=",")
                source.seek(0)
                return pd.read_csv(source, encoding=enc, thousands=",")
            except UnicodeDecodeError:
                continue
            except Exception as exc:
                raise ParseError(f"Failed to read CSV: {exc}") from exc
        raise ParseError("Cannot decode file — tried utf-8, latin-1, cp1252.")

    @staticmethod
    def _parse_amount(series: pd.Series) -> pd.Series:
        """Strip currency symbols and commas, convert to float."""
        cleaned = series.astype(str).str.replace(r"[$,\s]", "", regex=True)
        cleaned = cleaned.str.replace(r"\((.+)\)", r"-\1", regex=True)
        return pd.to_numeric(cleaned, errors="coerce")
