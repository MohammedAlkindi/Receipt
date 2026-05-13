"""Full pipeline integration test — no mocks, no API key required."""

from __future__ import annotations

from pathlib import Path


def test_full_pipeline_no_api(tmp_path: Path) -> None:
    """Run Chase sample through every layer except narrative generation."""
    from receipt.analysis.anomalies import AnomalyDetector
    from receipt.analysis.patterns import detect_patterns
    from receipt.ingestion.chase import ChaseParser
    from receipt.pipeline.aggregator import compute_stats
    from receipt.pipeline.categorizer import SemanticCategorizer
    from receipt.pipeline.cleaner import deduplicate, normalize_dates, normalize_descriptions
    from receipt.storage.store import ReceiptStore

    sample = Path(__file__).parent.parent / "data" / "sample" / "chase_sample.csv"

    # Ingest
    df = ChaseParser().parse(sample)
    assert len(df) == 30

    # Pipeline
    df = normalize_descriptions(df)
    df = deduplicate(df)
    df = normalize_dates(df)
    df = SemanticCategorizer(use_embeddings=False).categorize(df)
    df = AnomalyDetector().fit_predict(df)

    # Analysis
    stats = compute_stats(df)
    patterns = detect_patterns(df)
    assert isinstance(patterns, list)
    assert stats["total_spent"] < 0
    assert stats["total_income"] > 0
    assert "by_category" in stats

    # Verify CHANGE 10: by_category must not include raw transaction lists
    if stats["by_category"]:
        first_cat = next(iter(stats["by_category"].values()))
        assert "transactions" not in first_cat

    # Storage round-trip
    store = ReceiptStore(db_path=tmp_path / "test.db")
    run_id = store.save_analysis(
        period_start=df["date"].min().to_pydatetime(),
        period_end=df["date"].max().to_pydatetime(),
        transaction_count=len(df),
    )
    saved = store.save_transactions(df, run_id)
    assert saved == 30

    # Retrieval
    retrieved = store.get_transactions()
    assert len(retrieved) == 30

    # Second save must deduplicate (return 0 new rows)
    saved_again = store.save_transactions(df, run_id)
    assert saved_again == 0

    # Merchant upsert
    store.upsert_merchants(df)
    merchants = store.get_merchants()
    assert len(merchants) > 0

    # History
    history = store.get_analysis_history()
    assert len(history) == 1
    assert history[0]["run_id"] == run_id
