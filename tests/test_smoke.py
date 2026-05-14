"""Quick end-to-end smoke test (no API key needed)."""
from pathlib import Path

from receipt.analysis.anomalies import AnomalyDetector
from receipt.analysis.patterns import detect_patterns
from receipt.ingestion.chase import ChaseParser
from receipt.pipeline.aggregator import compute_stats
from receipt.pipeline.categorizer import SemanticCategorizer
from receipt.pipeline.cleaner import deduplicate, normalize_dates, normalize_descriptions


def test_pipeline_smoke():
    df = ChaseParser().parse(Path(__file__).parent.parent / "data" / "sample" / "chase_sample.csv")
    assert len(df) > 0

    df = normalize_descriptions(df)
    df = deduplicate(df)
    df = normalize_dates(df)
    df = SemanticCategorizer(use_embeddings=False).categorize(df)
    df = AnomalyDetector().fit_predict(df)

    stats = compute_stats(df)
    patterns = detect_patterns(df)

    assert "total_spent" in stats
    assert "total_income" in stats
    assert "net" in stats
    assert isinstance(patterns, list)
    assert "is_anomaly" in df.columns
