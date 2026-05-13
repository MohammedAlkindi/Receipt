"""Analysis layer — pattern detection, anomalies, narrative generation."""

from receipt.analysis.anomalies import AnomalyDetector
from receipt.analysis.narrator import NarrativeReport, Narrator
from receipt.analysis.patterns import Pattern, detect_patterns

__all__ = [
    "detect_patterns",
    "Pattern",
    "AnomalyDetector",
    "Narrator",
    "NarrativeReport",
]
