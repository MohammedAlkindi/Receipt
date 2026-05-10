"""Analysis layer — pattern detection, anomalies, narrative generation."""

from receipt.analysis.patterns import detect_patterns, Pattern
from receipt.analysis.anomalies import AnomalyDetector
from receipt.analysis.narrator import Narrator, NarrativeReport

__all__ = [
    "detect_patterns",
    "Pattern",
    "AnomalyDetector",
    "Narrator",
    "NarrativeReport",
]
