"""Pipeline layer — cleaning, categorization, aggregation, drift."""

from receipt.pipeline.cleaner import normalize_descriptions, deduplicate, normalize_dates
from receipt.pipeline.categorizer import SemanticCategorizer
from receipt.pipeline.aggregator import compute_stats
from receipt.pipeline.drift import DriftDetector, DriftReport

__all__ = [
    "normalize_descriptions",
    "deduplicate",
    "normalize_dates",
    "SemanticCategorizer",
    "compute_stats",
    "DriftDetector",
    "DriftReport",
]
