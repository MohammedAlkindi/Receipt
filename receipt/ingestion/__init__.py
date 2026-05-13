"""Ingestion layer — parsers and auto-detection factory."""

from pathlib import Path

from receipt.ingestion.base import ParseError, TransactionParser
from receipt.ingestion.bofa import BofAParser
from receipt.ingestion.chase import ChaseParser
from receipt.ingestion.csv_parser import GenericCSVParser
from receipt.ingestion.plaid import PlaidParser


def detect_parser(path: Path) -> TransactionParser:
    """Return the most appropriate parser for the given CSV file.

    Tries each bank-specific parser first; falls back to GenericCSVParser.
    Raises ParseError if the file cannot be read at all.
    """
    import pandas as pd

    try:
        sample = pd.read_csv(path, nrows=5, encoding="utf-8-sig")
    except Exception as exc:
        raise ParseError(f"Cannot read {path}: {exc}") from exc

    for parser_cls in (ChaseParser, BofAParser, PlaidParser):
        if parser_cls.detect(sample):
            return parser_cls()

    return GenericCSVParser()


__all__ = [
    "TransactionParser",
    "ParseError",
    "GenericCSVParser",
    "ChaseParser",
    "BofAParser",
    "PlaidParser",
    "detect_parser",
]
