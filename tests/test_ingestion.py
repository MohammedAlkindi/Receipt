"""Tests for the ingestion layer."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from receipt.ingestion import detect_parser
from receipt.ingestion.base import ParseError
from receipt.ingestion.bofa import BofAParser
from receipt.ingestion.chase import ChaseParser
from receipt.ingestion.csv_parser import GenericCSVParser
from receipt.ingestion.plaid import PlaidParser

_SAMPLE = Path(__file__).parent.parent / "data" / "sample"
_STANDARD_COLS = {"date", "description", "amount", "raw_description", "source", "transaction_id"}


# ---------------------------------------------------------------------------
# GenericCSVParser
# ---------------------------------------------------------------------------

class TestGenericCSVParser:
    def test_parses_standard_columns(self):
        csv = "date,description,amount\n2026-01-01,Coffee,-4.50\n2026-01-02,Salary,2000.00\n"
        parser = GenericCSVParser()
        df = parser.parse(io.StringIO(csv))
        assert _STANDARD_COLS.issubset(set(df.columns))
        assert len(df) == 2

    def test_detects_alternate_column_names(self):
        csv = "Transaction Date,Memo,Transaction Amount\n2026-01-01,Uber,-12.00\n"
        df = GenericCSVParser().parse(io.StringIO(csv))
        assert "description" in df.columns
        assert abs(df["amount"].iloc[0]) == 12.0

    def test_split_debit_credit_columns(self):
        csv = "date,description,debit,credit\n2026-01-01,Rent,1200.00,\n2026-01-02,Salary,,3000.00\n"
        df = GenericCSVParser().parse(io.StringIO(csv))
        assert df["amount"].iloc[0] < 0   # debit is negative
        assert df["amount"].iloc[1] > 0   # credit is positive

    def test_raises_parse_error_on_missing_date(self):
        csv = "description,amount\nCoffee,-4.50\n"
        with pytest.raises(ParseError, match="date"):
            GenericCSVParser().parse(io.StringIO(csv))

    def test_strips_currency_symbols(self):
        csv = "date,description,amount\n2026-01-01,Amazon,$34.99\n"
        df = GenericCSVParser().parse(io.StringIO(csv))
        assert df["amount"].iloc[0] == pytest.approx(34.99)

    def test_transaction_id_is_deterministic(self):
        csv = "date,description,amount\n2026-01-01,Coffee,-4.50\n"
        df1 = GenericCSVParser().parse(io.StringIO(csv))
        df2 = GenericCSVParser().parse(io.StringIO(csv))
        assert df1["transaction_id"].iloc[0] == df2["transaction_id"].iloc[0]

    def test_amount_takes_precedence_over_debit_credit_logs_warning(self, caplog):
        """Task 4: CSV with both amount and debit/credit logs a warning."""
        import logging

        csv = "date,description,amount,debit,credit\n2026-01-01,Coffee,-4.50,4.50,\n"
        with caplog.at_level(logging.WARNING, logger="receipt.ingestion.csv_parser"):
            df = GenericCSVParser().parse(io.StringIO(csv))
        assert df["amount"].iloc[0] == pytest.approx(-4.50)
        assert any(
            "amount" in rec.message and "takes precedence" in rec.message
            for rec in caplog.records
        )

    def test_zero_row_csv_raises_parse_error(self):
        """Task 4: A CSV with only a header raises ParseError."""
        csv = "date,description,amount\n"
        with pytest.raises(ParseError, match="no data rows"):
            GenericCSVParser().parse(io.StringIO(csv))

    def test_exact_match_before_partial_match(self):
        """Task 4: 'Transaction Date' exact match preferred over 'date_of_posting' partial."""
        # 'Transaction Date' is in _DATE_VARIANTS (exact); 'date_of_posting' would only
        # match as a partial. Exact match must win.
        csv = "date_of_posting,Transaction Date,description,amount\n2026-01-15,2026-01-01,Coffee,-4.50\n"
        df = GenericCSVParser().parse(io.StringIO(csv))
        # The 'Transaction Date' column has 2026-01-01 — confirm that date was used
        assert df["date"].iloc[0].day == 1


# ---------------------------------------------------------------------------
# ChaseParser
# ---------------------------------------------------------------------------

class TestChaseParser:
    def test_detects_chase_format(self):
        sample = pd.read_csv(_SAMPLE / "chase_sample.csv", nrows=2)
        assert ChaseParser.detect(sample)

    def test_does_not_detect_bofa_format(self):
        sample = pd.read_csv(_SAMPLE / "bofa_sample.csv", nrows=2)
        assert not ChaseParser.detect(sample)

    def test_parses_chase_sample(self, chase_df):
        assert _STANDARD_COLS.issubset(set(chase_df.columns))
        assert len(chase_df) == 30
        # Income rows should be positive
        income = chase_df[chase_df["amount"] > 0]
        assert len(income) >= 1

    def test_amounts_are_signed(self, chase_df):
        expenses = chase_df[chase_df["amount"] < 0]
        assert len(expenses) > 0

    def test_raises_on_bad_input(self):
        with pytest.raises(ParseError):
            ChaseParser().parse(io.StringIO("not,valid,csv\nfor,chase,bank\n"))


# ---------------------------------------------------------------------------
# BofAParser
# ---------------------------------------------------------------------------

class TestBofAParser:
    def test_detects_bofa_format(self):
        sample = pd.read_csv(_SAMPLE / "bofa_sample.csv", nrows=2)
        assert BofAParser.detect(sample)

    def test_does_not_detect_chase_format(self):
        sample = pd.read_csv(_SAMPLE / "chase_sample.csv", nrows=2)
        assert not BofAParser.detect(sample)

    def test_parses_bofa_sample(self, bofa_df):
        assert _STANDARD_COLS.issubset(set(bofa_df.columns))
        assert len(bofa_df) == 30

    def test_date_parsing(self, bofa_df):
        assert bofa_df["date"].dt.year.unique()[0] == 2026


# ---------------------------------------------------------------------------
# PlaidParser
# ---------------------------------------------------------------------------

class TestPlaidParser:
    def test_detects_plaid_format(self):
        sample = pd.read_csv(_SAMPLE / "plaid_sample.csv", nrows=2)
        assert PlaidParser.detect(sample)

    def test_inverts_sign(self, plaid_df):
        # Plaid positive → our negative (expense)
        expenses = plaid_df[plaid_df["amount"] < 0]
        assert len(expenses) > 0

    def test_parses_plaid_sample(self, plaid_df):
        assert _STANDARD_COLS.issubset(set(plaid_df.columns))


# ---------------------------------------------------------------------------
# Auto-detection factory
# ---------------------------------------------------------------------------

class TestDetectParser:
    def test_detects_chase(self):
        parser = detect_parser(_SAMPLE / "chase_sample.csv")
        assert isinstance(parser, ChaseParser)

    def test_detects_bofa(self):
        parser = detect_parser(_SAMPLE / "bofa_sample.csv")
        assert isinstance(parser, BofAParser)

    def test_detects_plaid(self):
        parser = detect_parser(_SAMPLE / "plaid_sample.csv")
        assert isinstance(parser, PlaidParser)

    def test_falls_back_to_generic(self, tmp_path):
        csv = tmp_path / "unknown.csv"
        csv.write_text("date,description,amount\n2026-01-01,Test,-5.00\n")
        parser = detect_parser(csv)
        assert isinstance(parser, GenericCSVParser)

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(ParseError):
            detect_parser(tmp_path / "nonexistent.csv")


# ---------------------------------------------------------------------------
# GenericCSVParser — sample file integration tests
# ---------------------------------------------------------------------------

class TestGenericSampleFile:
    def test_parses_generic_sample(self, generic_df):
        assert _STANDARD_COLS.issubset(set(generic_df.columns))
        assert len(generic_df) == 75

    def test_income_rows_are_positive(self, generic_df):
        assert len(generic_df[generic_df["amount"] > 0]) >= 1

    def test_expenses_are_negative(self, generic_df):
        assert len(generic_df[generic_df["amount"] < 0]) >= 1

    def test_date_range_spans_march_to_may(self, generic_df):
        assert generic_df["date"].dt.month.min() == 3
        assert generic_df["date"].dt.month.max() == 5

    def test_auto_detection_returns_generic_parser(self):
        result = detect_parser(_SAMPLE / "generic_sample.csv")
        assert isinstance(result, GenericCSVParser)
