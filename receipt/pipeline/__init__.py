"""Pipeline layer — cleaning, categorization, aggregation, drift."""

from receipt.pipeline.aggregator import compute_stats
from receipt.pipeline.audit import AuditLogger, PipelineAuditLog, StageEvent
from receipt.pipeline.categorizer import SemanticCategorizer
from receipt.pipeline.cleaner import deduplicate, normalize_dates, normalize_descriptions
from receipt.pipeline.drift import DriftDetector, DriftReport

__all__ = [
    "normalize_descriptions",
    "deduplicate",
    "normalize_dates",
    "SemanticCategorizer",
    "compute_stats",
    "DriftDetector",
    "DriftReport",
    "AuditLogger",
    "PipelineAuditLog",
    "StageEvent",
]
