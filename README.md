# receipt

[![CI](https://github.com/alkindymhmd/receipt/actions/workflows/ci.yml/badge.svg)](https://github.com/alkindymhmd/receipt/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](https://github.com/alkindymhmd/receipt/pulls)

Your bank app tells you how much you spent. **receipt** tells you what actually happened — semantic clustering, seven behavioral detectors, and an AI narrative that thinks like a sharp friend with an accounting degree: specific dollar amounts, named merchants, non-obvious patterns, and none of the generic budget-app moralizing.

---

## Architecture

```
CSV file / base64 payload
         │
         ▼
 ┌───────────────┐
 │   Ingestion   │  Auto-detects bank format; Chase / BofA / Plaid / Generic
 │  (parsers)    │  → standard DataFrame: {date, description, amount, source, tx_id}
 └──────┬────────┘
        │
        ▼
 ┌───────────────┐
 │   Pipeline    │
 │               │  1. cleaner    — strip noise, normalise names, deduplicate
 │               │  2. categorizer— sentence-transformers + HDBSCAN + keyword fallback
 │               │  3. aggregator — totals, by-category, by-week, by-merchant
 │               │  4. drift      — compare to previous period; velocity + category change
 └──────┬────────┘
        │
        ▼
 ┌───────────────┐
 │   Analysis    │
 │               │  1. patterns   — 7 heuristic detectors (subscriptions, anomalous weeks…)
 │               │  2. anomalies  — Isolation Forest scores every transaction
 │               │  3. narrator   — Anthropic API → JSON NarrativeReport
 └──────┬────────┘
        │
        ▼
 ┌───────────────┐
 │   Storage     │  SQLite via SQLAlchemy (Transactions, AnalysisRuns, Merchants)
 └──────┬────────┘
        │
        ├──▶ CLI (Typer + rich)
        └──▶ API (FastAPI + uvicorn)
```

---

## Features

- **Auto-detects bank format** from column signatures — drop a Chase, BofA, Plaid, or generic CSV export and the right parser is selected without a flag.
- **Semantic transaction clustering** via `all-MiniLM-L6-v2` + HDBSCAN groups transactions by meaning, not keyword matching — "SBUX #04892" and "Starbucks Store Times Sq" land in the same cluster automatically.
- **Seven behavioral detectors** flag subscription creep, anomalous spending weeks, weekend splurge, single-merchant dominance, late-night food spend, income irregularity, and recurring-forgotten merchants.
- **Isolation Forest anomaly scoring** assigns an outlier score to every individual transaction, surfacing charges that are statistically unusual relative to your own patterns.
- **Month-over-month drift tracking** reports which categories accelerated or shrank >20%, which merchants appeared or disappeared, and whether second-half spend outpaced first-half.
- **Forensic AI narrative** via the Anthropic API generates a TL;DR, 3–5 insights anchored to specific merchants and dollar amounts, and a concrete next-steps paragraph — with a quality guard that suppresses generic phrases before the report is returned.
- **Persistent run history** stores every analysis to SQLite; replay, compare, or export any past run as Markdown with a single command.
- **CLI and REST API** — the same pipeline runs as a `receipt analyze` command or as a FastAPI server accepting base64-encoded CSV payloads for programmatic access.

---

## Installation

```bash
# Clone and install in editable mode
git clone https://github.com/alkindymhmd/receipt
cd receipt
pip install -e .

# For development tools (pytest, ruff, mypy)
pip install -e ".[dev]"

# Copy and fill in your API key
cp .env.example .env
# Set ANTHROPIC_API_KEY=sk-ant-...
```

> **Python ≥ 3.11 required.**

> **Note:** The first run downloads the `all-MiniLM-L6-v2` sentence-transformers model (~80 MB).
> Use `receipt demo` or pass `use_embeddings=False` to skip this download.

---

## Database Migrations

receipt uses [Alembic](https://alembic.sqlalchemy.org/) to manage schema migrations. You must run migrations once after installation and again after each upgrade:

```bash
# After first install
alembic upgrade head

# After upgrading receipt (e.g. via pip install --upgrade receipt)
alembic upgrade head
```

By default the database lives at `~/.receipt/receipt.db`. Override this with the `RECEIPT_DB_PATH` environment variable:

```bash
RECEIPT_DB_PATH=/data/my_finances.db alembic upgrade head
```

Check which revision is currently applied:

```bash
alembic history          # show all revisions
alembic current          # show applied revision
```

---

## Quick Start

```bash
# Step 1: install (required before any receipt command works)
pip install -e .

# Step 2: apply database schema (required once after install)
alembic upgrade head

# Try the bundled Chase sample — no API key, no model download
receipt demo

# Analyse your own file with auto-detection
receipt analyze ~/Downloads/chase_activity.csv

# Full run: compare to last month, save to DB, generate AI narrative
receipt analyze ~/Downloads/chase_activity.csv \
  --compare \
  --save \
  --period 30 \
  --api-key $ANTHROPIC_API_KEY
```

Expected terminal output (illustrative — numbers match the bundled Chase sample, narrative text is representative):

```
╭──────────────────────────────────────────────────────╮
│                   Spending Summary                   │
├─────────────────────────────┬────────────────────────┤
│ Metric                      │ Value                  │
├─────────────────────────────┼────────────────────────┤
│ Transactions                │ 29                     │
│ Total Spent                 │ $874.23                │
│ Total Income                │ $3,200.00              │
│ Net                         │ $2,325.77              │
│ Subscriptions               │ $117.46                │
│ Most Visited                │ Trader Joe's (2×)      │
│ Largest Purchase            │ $149.00 at Amazon      │
╰─────────────────────────────┴────────────────────────╯

╭─────────────────── SUBSCRIPTION_CREEP ───────────────╮
│  6 active subscriptions totalling $117.46            │
╰──────────────────────────────────────────────────────╯

╭──────────────────── AI Insights ─────────────────────╮
│                                                      │
│  TL;DR  Delivery apps captured 80% of your dining   │
│         budget: $122 of $152 spent eating out went  │
│         to Uber Eats, DoorDash, or Grubhub.         │
│                                                      │
│  ▸ Grubhub's April 19 order ($38.50) was your       │
│    third-largest individual charge this month —      │
│    behind only rent and the $149 Amazon purchase.   │
│    Four delivery orders in 19 days is a run rate    │
│    of ~$237/month if April's frequency holds.       │
│                                                      │
╰──────────────────────────────────────────────────────╯
```

---

## CLI Reference

### `receipt demo`

Run the full pipeline on the bundled Chase sample (30 transactions, April 2026). No API key or internet connection required — embeddings are disabled for speed.

```bash
receipt demo
```

### `receipt analyze <FILE> [OPTIONS]`

| Option | Default | Description |
|---|---|---|
| `--format` / `-f` | `auto` | `auto` \| `chase` \| `bofa` \| `plaid` \| `generic` |
| `--period` / `-p` | `30` | Days to analyse (30 / 60 / 90) |
| `--compare` | off | Compare to previous stored period |
| `--output` / `-o` | `terminal` | `terminal` \| `json` \| `markdown` |
| `--save` | off | Save results to `~/.receipt/receipt.db` |
| `--api-key` | env | Anthropic API key (falls back to `ANTHROPIC_API_KEY`) |

```bash
# JSON output — pipe to jq
receipt analyze transactions.csv --output json | jq '.narrative.tldr'

# Markdown report — redirect to file
receipt analyze transactions.csv --output markdown > report.md

# Force BofA parser
receipt analyze export.csv --format bofa --period 60
```

### `receipt history`

List past analysis runs with dates and AI summaries.

```bash
$ receipt history
┌──────────────┬────────────┬──────────────────────────┬───────┬──────────────────────────────────┐
│ Run ID       │ Date       │ Period                   │ Txns  │ TL;DR                            │
├──────────────┼────────────┼──────────────────────────┼───────┼──────────────────────────────────┤
│ a3f9c2e1b0d4 │ 2026-05-10 │ 2026-04-01 – 2026-04-30  │  29   │ Food delivery ate 34% of...      │
└──────────────┴────────────┴──────────────────────────┴───────┴──────────────────────────────────┘
```

### `receipt merchants`

Top merchants by lifetime spend (requires `--save` on at least one prior run).

### `receipt export [RUN_ID]`

Export a past run as Markdown to stdout:

```bash
receipt export a3f9c2e1b0d4 > april-report.md
```

### `receipt serve`

Start the FastAPI server:

```bash
receipt serve --port 8000
# Swagger UI at http://localhost:8000/docs
```

---

## API Usage

### `POST /analyze`

```bash
# Encode your CSV and call the endpoint
CSV_B64=$(base64 -w 0 chase_activity.csv)

curl -s http://localhost:8000/analyze \
  -H "X-Api-Key: $ANTHROPIC_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"file_content\": \"$CSV_B64\", \"format\": \"chase\", \"period\": 30}" \
  | jq '.narrative.tldr'
```

### `GET /history`

```bash
curl http://localhost:8000/history | jq '.[0].tldr'
```

### `GET /history/{run_id}`

```bash
curl http://localhost:8000/history/a3f9c2e1b0d4 | jq '.narrative'
```

### `GET /merchants`

```bash
curl http://localhost:8000/merchants | jq '.[0:5]'
```

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status": "ok", "db": {"transactions": 29, "analysis_runs": 1, "merchants": 18}, "narrative_service": "unknown"}
```

---

## How the Semantic Categorizer Works

1. **Embedding** — Each transaction description is passed through `all-MiniLM-L6-v2` (a 22M-parameter sentence-transformer model, ~80 MB). This converts the raw string into a 384-dimensional dense vector that captures semantic meaning, so "SBUX #04892" and "Starbucks Store Times Sq" land near each other in vector space.

2. **Clustering** — HDBSCAN groups all transaction vectors into variable-density clusters. Clusters are determined by the actual shape of your data, not a preset number of categories k. Outliers get `cluster_id = -1`.

3. **Labelling** — Each cluster is matched to a human-readable category by computing cosine similarity between the cluster centroid and pre-computed centroids for 8 seed categories (`food_dining`, `groceries`, `subscriptions`, `transportation`, `shopping`, `income`, `health`, `housing`). Seed centroids are averaged embeddings of a curated vocabulary list.

4. **Fallback** — If `sentence-transformers` is unavailable (e.g., CI environment), the categorizer falls back to fast keyword matching against the same seed vocabulary with a fixed 0.9 confidence score.

Embeddings and HDBSCAN labels are not cached between runs by default; call `cache_embeddings(path)` to persist the seed centroids to disk and skip re-encoding on subsequent runs.

---

## How Drift Detection Works

`DriftDetector.compare_periods(df_current, df_previous)` produces a `DriftReport`:

| Field | Description |
|---|---|
| `increased` | Categories that grew >20% vs prior period |
| `decreased` | Categories that shrank >20% |
| `new_merchants` | Merchants in current period not seen before |
| `dropped_merchants` | Merchants in previous period that disappeared |
| `velocity_trend` | `accelerating` if second-half spend >1.25× first-half |
| `subscription_drift` | New / cancelled subscription merchants |
| `narrative_hints` | Pre-formed plain-English sentences fed to the narrator |

The 20% threshold is configurable via `DriftDetector.DRIFT_THRESHOLD`.

---

## Configuration

All settings can be placed in `.env` at the project root:

```env
ANTHROPIC_API_KEY=sk-ant-...
RECEIPT_DB_PATH=~/.receipt/receipt.db
RECEIPT_API_HOST=0.0.0.0
RECEIPT_API_PORT=8000
RECEIPT_MODEL_CACHE=~/.receipt/models
RECEIPT_LOG_LEVEL=INFO
```

---

## Adding a New Bank Parser

1. Create `receipt/ingestion/mybank.py`:

```python
from receipt.ingestion.base import ParseError, TransactionParser
import pandas as pd

_MYBANK_COLS = {"Trans Date", "Description", "Amount"}

class MyBankParser(TransactionParser):
    @classmethod
    def detect(cls, df: pd.DataFrame) -> bool:
        return _MYBANK_COLS.issubset({c.strip() for c in df.columns})

    def parse(self, source):
        raw = pd.read_csv(source, encoding="utf-8-sig")
        raw.columns = [c.strip() for c in raw.columns]
        result = pd.DataFrame()
        result["date"] = raw["Trans Date"]
        result["description"] = raw["Description"].fillna("").astype(str)
        result["raw_description"] = result["description"]
        result["amount"] = pd.to_numeric(raw["Amount"], errors="coerce")
        return self._finalise(result, source_name="mybank")
```

2. Register it in `receipt/ingestion/__init__.py`:

```python
from receipt.ingestion.mybank import MyBankParser

def detect_parser(path):
    ...
    for parser_cls in (ChaseParser, BofAParser, PlaidParser, MyBankParser):
        if parser_cls.detect(sample):
            return parser_cls()
    return GenericCSVParser()
```

3. Add a `@classmethod detect` that returns `True` only for your bank's specific column signature — this prevents false positives when the factory tries every parser.

4. Add a fixture and tests in `tests/test_ingestion.py` mirroring the existing bank tests.

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=receipt --cov-report=term-missing

# Run a single module
pytest tests/test_ingestion.py -v

# Skip slow embedding tests (avoids ~80MB model download in CI)
pytest -k "not embedding"
```

Tests use the sample CSVs in `data/sample/` as fixtures and do **not** require an Anthropic API key — the narrator tests mock the API call.

---

## Troubleshooting

**`receipt` command not found after installation**

Ensure your Python `Scripts` directory is in `PATH`, or invoke the CLI directly:

```bash
python -m receipt.cli --help
```

On Windows, the Scripts directory is typically `%APPDATA%\Python\PythonXY\Scripts`. On macOS/Linux it is `~/.local/bin` (user install) or inside your virtualenv's `bin/`.

**First run is slow / downloads a large file**

`receipt analyze` downloads the `all-MiniLM-L6-v2` model (~80 MB) on first use. Use `receipt demo` to skip the download entirely, or set `use_embeddings=False` in the categorizer for local testing.

**Narrative generation times out**

The Anthropic API call has a 30-second timeout. If your network is slow or the API is degraded, you will see a `502` from the server or a warning in the CLI. Check `https://status.anthropic.com` and retry.

---

## Roadmap

- [ ] **PyPI release** — package is install-from-source only; publishing to PyPI will enable `pip install receipt` without cloning.
- [ ] **Hosted demo** — a zero-install web interface where anyone can upload a CSV and see an analysis without running a local server.
- [ ] **Database migration system** — schema changes currently require deleting `~/.receipt/receipt.db`; Alembic migrations would make upgrades non-destructive.
- [ ] **Multi-user API support** — the FastAPI server has no authentication or user isolation; supporting multiple users requires API key auth and per-user database partitioning.

---

## Known Limitations

- **Max 50,000 transactions per file** — the generic CSV parser enforces this guard to prevent memory exhaustion.
- **sentence-transformers requires ~80 MB download on first run** — use `SemanticCategorizer(use_embeddings=False)` or `receipt demo` to skip.
- **HDBSCAN clusters are most meaningful on 200+ transactions** — on smaller datasets the keyword fallback produces better category labels.
- **FastAPI server is single-process** — for concurrent use, run with multiple workers: `uvicorn receipt.api.server:app --workers 4` (requires `gunicorn` as process manager on Linux/macOS).
- **No database migration system** — schema changes require deleting `~/.receipt/receipt.db` and re-running analyses.

---

## License

MIT — see `pyproject.toml`.
