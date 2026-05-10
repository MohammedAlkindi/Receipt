"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

_SAMPLE = Path(__file__).parent.parent / "data" / "sample"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal well-formed standard-schema DataFrame."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-04-01",
                    "2026-04-05",
                    "2026-04-10",
                    "2026-04-15",
                    "2026-04-20",
                ]
            ).tz_localize("UTC"),
            "description": [
                "Starbucks",
                "Trader Joe'S",
                "Netflix",
                "Uber Eats",
                "Payroll Deposit",
            ],
            "amount": [-6.45, -87.32, -15.49, -24.67, 3200.0],
            "raw_description": [
                "STARBUCKS #04892",
                "TRADER JOE S 142",
                "NETFLIX.COM",
                "UBER* EATS 3948201",
                "DIRECT DEPOSIT PAYROLL",
            ],
            "source": ["chase"] * 5,
            "transaction_id": [f"abc{i:013d}" for i in range(5)],
        }
    )


@pytest.fixture
def chase_df() -> pd.DataFrame:
    from receipt.ingestion.chase import ChaseParser

    return ChaseParser().parse(_SAMPLE / "chase_sample.csv")


@pytest.fixture
def bofa_df() -> pd.DataFrame:
    from receipt.ingestion.bofa import BofAParser

    return BofAParser().parse(_SAMPLE / "bofa_sample.csv")


@pytest.fixture
def plaid_df() -> pd.DataFrame:
    from receipt.ingestion.plaid import PlaidParser

    return PlaidParser().parse(_SAMPLE / "plaid_sample.csv")
