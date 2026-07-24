"""Generic CSV parser with automatic column detection."""

from __future__ import annotations

import logging
import re
from io import IOBase
from pathlib import Path
from typing import Literal, Union

import pandas as pd

from receipt.ingestion.base import MAX_ROWS, ParseError, TransactionParser

logger = logging.getLogger(__name__)

# Known column name variants for fuzzy matching
_DATE_VARIANTS = {"date", "transaction date", "posted date", "trans date", "post date"}
_DESC_VARIANTS = {"description", "memo", "details", "payee", "merchant", "name", "transaction"}
_AMOUNT_VARIANTS = {"amount", "transaction amount", "trans amount"}
_DEBIT_VARIANTS = {"debit", "withdrawal", "debit amount", "withdrawals"}
_CREDIT_VARIANTS = {"credit", "deposit", "credit amount", "deposits"}


def _normalise_col(col: str) -> str:
    return re.sub(r"\s+", " ", col.strip().lower())


def _find_col(
    columns: list[str],
    variants: set[str],
    priority: Literal["exact", "partial"] = "exact",
) -> str | None:
    """Return the first column matching *variants*.

    Tries exact normalized matches first, then falls back to partial (substring)
    matches regardless of the *priority* parameter — callers rely on this ordering
    to prefer e.g. "Transaction Date" over "date_of_posting".
    """
    norm = {_normalise_col(c): c for c in columns}

    # Exact pass: normalized column name must be in variants
    for variant in sorted(variants):  # sorted for deterministic ordering
        if variant in norm:
            return norm[variant]

    if priority == "exact":
        return None

    # Partial pass: variant appears as substring of the normalized column name
    for col_norm, col_orig in norm.items():
        for variant in sorted(variants):
            if variant in col_norm:
                return col_orig
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

        if len(df) == 0:
            raise ParseError("CSV has no data rows")

        cols = list(df.columns)

        for warning in self.detect_ambiguities(cols):
            logger.warning("GenericCSVParser: %s", warning)

        date_col = _find_col(cols, _DATE_VARIANTS)
        desc_col = _find_col(cols, _DESC_VARIANTS)
        amount_col = _find_col(cols, _AMOUNT_VARIANTS)
        debit_col = _find_col(cols, _DEBIT_VARIANTS)
        credit_col = _find_col(cols, _CREDIT_VARIANTS)

        # Conflict: amount AND debit/credit present — amount takes precedence
        if amount_col and (debit_col or credit_col):
            logger.warning(
                "GenericCSVParser: both 'amount' and debit/credit columns detected; "
                "'amount' takes precedence. Ignoring debit/credit columns."
            )

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

    @staticmethod
    def detect_ambiguities(columns: list[str]) -> list[str]:
        """Return human-readable warnings about ambiguous column sets."""
        warnings: list[str] = []
        norm_cols = [_normalise_col(c) for c in columns]

        # Multiple date candidates
        date_matches = [c for c in norm_cols if any(v in c for v in _DATE_VARIANTS)]
        if len(date_matches) > 1:
            warnings.append(
                f"Multiple date-like columns detected: {date_matches}. "
                "Using first exact match."
            )

        # Multiple amount candidates
        amount_matches = [c for c in norm_cols if any(v in c for v in _AMOUNT_VARIANTS)]
        if len(amount_matches) > 1:
            warnings.append(
                f"Multiple amount-like columns detected: {amount_matches}. "
                "Using first exact match."
            )

        return warnings

    def _read_raw(self, source: Union[str, Path, IOBase]) -> pd.DataFrame:
        encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                if isinstance(source, (str, Path)):
                    df = pd.read_csv(source, encoding=enc, thousands=",", nrows=MAX_ROWS + 1)
                else:
                    source.seek(0)
                    df = pd.read_csv(source, encoding=enc, thousands=",", nrows=MAX_ROWS + 1)
            except UnicodeDecodeError:
                continue
            except Exception as exc:
                raise ParseError(f"Failed to read CSV: {exc}") from exc

            if len(df) > MAX_ROWS:
                raise ParseError(
                    f"File has more than {MAX_ROWS} rows. Maximum supported is "
                    f"{MAX_ROWS}. Split your file into smaller chunks."
                )
            return df
        raise ParseError("Cannot decode file — tried utf-8, latin-1, cp1252.")

    @staticmethod
    def _parse_amount(series: pd.Series) -> pd.Series:
        """Strip currency symbols and commas, convert to float."""
        cleaned = series.astype(str).str.replace(r"[$,\s]", "", regex=True)
        cleaned = cleaned.str.replace(r"\((.+)\)", r"-\1", regex=True)
        return pd.to_numeric(cleaned, errors="coerce")
