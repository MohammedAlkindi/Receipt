"""Data normalization, deduplication, and date enrichment."""

from __future__ import annotations

import re

import pandas as pd

# Patterns that represent noise appended to merchant names
_NOISE_PATTERNS = [
    r"\s+\d{4,}",                  # trailing 4+ digit IDs: "UBER 4829482"
    r"\s*\*+\s*\d+",               # asterisk + digits: "UBER* 48294"
    r"\s*#\s*\d+",                 # hash + digits: "STARBUCKS #04892"
    r"\bORIG\s+CO\b.*",            # ACH originator: "ORIG CO EMPLOYER..."
    r"\bACH\b.*",                  # ACH references
    r"\bREF\s*#?\s*\d+\b.*",       # reference numbers
    r"\bCONF\s*#?\s*\w+\b.*",      # confirmation numbers
    r"\b\d{4}\s+\d{4}\s+\d{4}\b",  # partial card numbers
    r"\bONLINE\s+PURCHASE\b",
    r"\bPURCHASE\b",
    r"\bPAYMENT\b$",
    r"\bSALE\b$",
    r"\bTRANSACTION\b$",
    r"\s{2,}",                     # double spaces -> single
]

_MERCHANT_NORMALIZATIONS: list[tuple[str, str]] = [
    (r"UBER\s*\*?\s*EATS", "Uber Eats"),
    (r"UBER\s*\*?\s*TRIP|UBER\s+RIDES?|^UBER$", "Uber"),
    (r"LYFT\s*\*?\s*RIDE|^LYFT$", "Lyft"),
    (r"DOORDASH|DOOR\s*DASH", "DoorDash"),
    (r"GRUBHUB", "Grubhub"),
    (r"TRADER\s+JOE[S']?\s*\w*", "Trader Joe's"),
    (r"WHOLE\s+FOODS\s*\w*", "Whole Foods"),
    (r"STARBUCKS\s*\w*", "Starbucks"),
    (r"MCDONALD[S']?\s*\w*", "McDonald's"),
    (r"CHIPOTLE\s*\w*", "Chipotle"),
    (r"NETFLIX\s*\w*", "Netflix"),
    (r"SPOTIFY\s*\w*", "Spotify"),
    (r"HULU\s*\w*", "Hulu"),
    (r"ADOBE\s*\w*", "Adobe"),
    (r"AMAZON\s*\w*|AMZN\s*\w*", "Amazon"),
    (r"CHATGPT|OPENAI", "OpenAI ChatGPT"),
    (r"CVS\s*\w*", "CVS Pharmacy"),
    (r"WALGREENS\s*\w*", "Walgreens"),
    (r"TARGET\s*\w*", "Target"),
    (r"WALMART\s*\w*", "Walmart"),
]


def normalize_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with 'description' column cleaned and normalized."""
    df = df.copy()
    desc = df["description"].str.upper().str.strip()

    # Strip noise patterns
    for pattern in _NOISE_PATTERNS[:-1]:  # skip double-space (applied last)
        desc = desc.str.replace(pattern, "", regex=True, case=False)
    desc = desc.str.replace(r"\s{2,}", " ", regex=True).str.strip()

    # Apply merchant-level normalization
    for pattern, replacement in _MERCHANT_NORMALIZATIONS:
        mask = desc.str.match(r".*" + pattern + r".*", case=False, na=False)
        if mask.any():
            desc = desc.str.replace(pattern, replacement, regex=True, case=False)

    # Title-case anything not already normalized
    df["description"] = desc.str.title()
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicates by transaction_id; flag near-duplicates."""
    df = df.copy()

    # Exact deduplication
    df = df.drop_duplicates(subset=["transaction_id"], keep="first")

    # Near-duplicate detection: same merchant + same amount within 2 days
    df = df.sort_values("date").reset_index(drop=True)
    df["_near_dup"] = False

    for idx, row in df.iterrows():
        if df.at[idx, "_near_dup"]:
            continue
        window = df[
            (df["description"] == row["description"])
            & (df["amount"] == row["amount"])
            & (df["date"] >= row["date"])
            & (df["date"] <= row["date"] + pd.Timedelta(days=2))
            & (df.index != idx)
        ]
        if not window.empty:
            df.loc[window.index, "_near_dup"] = True

    df = df[~df["_near_dup"]].drop(columns=["_near_dup"])
    return df.reset_index(drop=True)


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal feature columns derived from the date field."""
    df = df.copy()
    dt = df["date"].dt
    df["day_of_week"] = dt.day_name()
    df["week_of_month"] = (dt.day - 1) // 7 + 1
    df["is_weekend"] = dt.dayofweek >= 5
    df["hour"] = dt.hour  # always 0 for date-only sources
    return df
