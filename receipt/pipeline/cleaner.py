"""Data normalization, deduplication, and date enrichment."""

from __future__ import annotations

import pandas as pd

from receipt.pipeline.audit import AuditLogger, PipelineAuditLog

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


def normalize_descriptions(
    df: pd.DataFrame,
    audit_log: PipelineAuditLog | None = None,
) -> pd.DataFrame:
    """Return df with 'description' column cleaned and normalized."""
    with AuditLogger(audit_log, "normalize_descriptions", len(df)) as al:
        df = df.copy()
        desc = df["description"].str.upper().str.strip()

        for pattern in _NOISE_PATTERNS[:-1]:  # skip double-space (applied last)
            desc = desc.str.replace(pattern, "", regex=True, case=False)
        desc = desc.str.replace(r"\s{2,}", " ", regex=True).str.strip()

        for pattern, replacement in _MERCHANT_NORMALIZATIONS:
            mask = desc.str.match(r".*" + pattern + r".*", case=False, na=False)
            if mask.any():
                desc = desc.str.replace(pattern, replacement, regex=True, case=False)

        df["description"] = desc.str.title()
        al.output_rows = len(df)
    return df


def deduplicate(
    df: pd.DataFrame,
    near_dup_window_days: int = 2,
    audit_log: PipelineAuditLog | None = None,
) -> pd.DataFrame:
    """Remove exact duplicates by transaction_id and vectorized near-duplicates.

    *near_dup_window_days* controls the near-duplicate detection window.
    Valid range: 0 (disable) to 7. Set to 0 to skip near-duplicate detection.
    """
    if not 0 <= near_dup_window_days <= 7:
        raise ValueError(
            f"near_dup_window_days must be between 0 and 7, got {near_dup_window_days}"
        )

    input_rows = len(df)
    with AuditLogger(audit_log, "deduplicate", input_rows) as al:
        df = df.copy()

        df = df.drop_duplicates(subset=["transaction_id"], keep="first")
        df = df.sort_values("date").reset_index(drop=True)

        if near_dup_window_days > 0:
            df["_key"] = df["description"] + "|" + df["amount"].astype(str)
            df["_prev_date"] = df.groupby("_key")["date"].shift(1)
            df["_gap"] = (df["date"] - df["_prev_date"]).dt.total_seconds() / 86400
            near_dup_mask = df["_gap"].notna() & (df["_gap"] >= 0) & (df["_gap"] <= near_dup_window_days)
            df = df[~near_dup_mask].drop(columns=["_key", "_prev_date", "_gap"])
            df = df.reset_index(drop=True)

        al.output_rows = len(df)
        al.metadata["duplicates_removed"] = input_rows - len(df)
    return df


def normalize_dates(
    df: pd.DataFrame,
    audit_log: PipelineAuditLog | None = None,
) -> pd.DataFrame:
    """Add temporal feature columns derived from the date field."""
    with AuditLogger(audit_log, "normalize_dates", len(df)) as al:
        df = df.copy()
        dt = df["date"].dt
        df["day_of_week"] = dt.day_name()
        df["week_of_month"] = (dt.day - 1) // 7 + 1
        df["is_weekend"] = dt.dayofweek >= 5
        df["hour"] = dt.hour  # always 0 for date-only sources
        al.output_rows = len(df)
    return df
