"""FastAPI REST server."""

from __future__ import annotations

import base64
import io
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

import anyio
from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from receipt import configure_logging


@asynccontextmanager
async def lifespan(app_instance: FastAPI):  # type: ignore[type-arg]
    configure_logging()
    yield


app = FastAPI(
    title="receipt API",
    description="Intelligent personal finance analysis engine.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_RECEIPT_TOKEN: str | None = os.getenv("RECEIPT_API_TOKEN")
_MAX_BODY_BYTES = 10 * 1024 * 1024  # 10 MB


@app.middleware("http")
async def limit_body_size(request: Request, call_next: Callable[..., Any]) -> Any:
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > _MAX_BODY_BYTES:
        return JSONResponse(
            status_code=413,
            content={"detail": "Request body too large. Maximum 10MB."},
        )
    return await call_next(request)


def _require_receipt_token(x_receipt_token: Optional[str] = Header(None)) -> None:
    if _RECEIPT_TOKEN and x_receipt_token != _RECEIPT_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Receipt-Token.",
        )


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    file_content: str  # base64-encoded CSV
    format: str = "auto"  # auto|chase|bofa|plaid|generic
    period: int = 30  # days


class InsightModel(BaseModel):
    headline: str
    detail: str


class NarrativeModel(BaseModel):
    tldr: str
    insights: list[InsightModel]
    next_steps: str
    generated_at: str


class PatternModel(BaseModel):
    type: str
    headline: str
    severity: str
    data: dict[str, Any]


class AnalyzeResponse(BaseModel):
    run_id: Optional[str]
    transaction_count: int
    stats: dict[str, Any]
    patterns: list[PatternModel]
    drift: Optional[dict[str, Any]]
    narrative: Optional[NarrativeModel]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
async def health() -> dict[str, Any]:
    """Return service status and DB stats."""
    try:
        from receipt.storage.store import ReceiptStore

        store = ReceiptStore()
        db_stats = store.db_stats()
        status_val = "ok"
    except Exception as exc:
        db_stats = {}
        status_val = f"degraded: {exc}"
    return {
        "status": status_val,
        "db": db_stats,
        "version": "0.1.0",
        "narrative_service": "unknown",
    }


@app.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
async def analyze(
    request: AnalyzeRequest,
    x_api_key: str = Header(..., description="Anthropic API key"),
) -> AnalyzeResponse:
    """Analyze a base64-encoded CSV and return a NarrativeReport."""
    try:
        csv_bytes = base64.b64decode(request.file_content)
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid base64 content.")

    if not x_api_key.startswith("sk-ant-"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid API key format. Anthropic keys must begin with 'sk-ant-'.",
        )

    csv_io = io.StringIO(csv_bytes.decode("utf-8-sig", errors="replace"))

    # Ingest
    from receipt.ingestion.bofa import BofAParser
    from receipt.ingestion.chase import ChaseParser
    from receipt.ingestion.csv_parser import GenericCSVParser
    from receipt.ingestion.plaid import PlaidParser

    parser_map = {
        "chase": ChaseParser,
        "bofa": BofAParser,
        "plaid": PlaidParser,
        "generic": GenericCSVParser,
    }

    try:
        if request.format == "auto":
            import pandas as pd

            sample = pd.read_csv(io.StringIO(csv_bytes.decode("utf-8-sig", errors="replace")), nrows=5)
            for cls in (ChaseParser, BofAParser, PlaidParser):
                if cls.detect(sample):
                    parser = cls()
                    break
            else:
                parser = GenericCSVParser()
            csv_io.seek(0)
        else:
            parser = parser_map.get(request.format, GenericCSVParser)()

        df = parser.parse(csv_io)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    # Filter period
    cutoff = datetime.now(timezone.utc) - timedelta(days=request.period)
    df = df[df["date"] >= cutoff].copy()
    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"No transactions found in the last {request.period} days.",
        )

    # Pipeline — all blocking calls offloaded to a thread pool
    from receipt.analysis.anomalies import AnomalyDetector
    from receipt.analysis.narrator import Narrator
    from receipt.analysis.patterns import detect_patterns
    from receipt.pipeline.aggregator import compute_stats
    from receipt.pipeline.categorizer import SemanticCategorizer
    from receipt.pipeline.cleaner import deduplicate, normalize_dates, normalize_descriptions

    _df = [df]
    _df[0] = await anyio.to_thread.run_sync(lambda: normalize_descriptions(_df[0]))
    _df[0] = await anyio.to_thread.run_sync(lambda: deduplicate(_df[0]))
    _df[0] = await anyio.to_thread.run_sync(lambda: normalize_dates(_df[0]))
    _df[0] = await anyio.to_thread.run_sync(lambda: SemanticCategorizer().categorize(_df[0]))
    _df[0] = await anyio.to_thread.run_sync(lambda: AnomalyDetector().fit_predict(_df[0]))
    df = _df[0]

    stats = await anyio.to_thread.run_sync(lambda: compute_stats(df))
    patterns = await anyio.to_thread.run_sync(lambda: detect_patterns(df))

    # Narrative
    narrator = Narrator(api_key=x_api_key)
    try:
        narrative_report = narrator.generate_narrative(stats, patterns)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Narrative generation failed: {exc}",
        )

    # Save
    from receipt.storage.store import ReceiptStore

    store = ReceiptStore()
    run_id = store.save_analysis(
        period_start=df["date"].min().to_pydatetime(),
        period_end=df["date"].max().to_pydatetime(),
        transaction_count=len(df),
        narrative=narrative_report.to_dict(),
    )
    store.save_transactions(df, run_id)
    store.upsert_merchants(df)

    narrative_model = NarrativeModel(
        tldr=narrative_report.tldr,
        insights=[InsightModel(headline=i.headline, detail=i.detail) for i in narrative_report.insights],
        next_steps=narrative_report.next_steps,
        generated_at=narrative_report.generated_at,
    )

    pattern_models = [
        PatternModel(type=p.type, headline=p.headline, severity=p.severity, data=p.data)
        for p in patterns
    ]

    return AnalyzeResponse(
        run_id=run_id,
        transaction_count=len(df),
        stats=stats,
        patterns=pattern_models,
        drift=None,
        narrative=narrative_model,
    )


@app.get("/history", tags=["history"])
async def get_history() -> list[dict[str, Any]]:
    """Return a list of past analysis run summaries."""
    from receipt.storage.store import ReceiptStore

    store = ReceiptStore()
    return await anyio.to_thread.run_sync(lambda: store.get_analysis_history())


@app.get("/history/{run_id}", tags=["history"])
async def get_run(run_id: str) -> dict[str, Any]:
    """Return the full details of a specific analysis run."""
    from receipt.storage.store import ReceiptStore

    store = ReceiptStore()
    run = store.get_analysis_run(run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")
    return run


@app.get("/merchants", tags=["data"])
async def get_merchants(limit: int = 30) -> list[dict[str, Any]]:
    """Return top merchants by total spend."""
    from receipt.storage.store import ReceiptStore

    store = ReceiptStore()
    return await anyio.to_thread.run_sync(lambda: store.get_merchants(limit=limit))
