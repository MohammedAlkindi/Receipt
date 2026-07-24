"""Semantic transaction categorizer using embeddings and HDBSCAN."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from receipt.pipeline.audit import PipelineAuditLog

logger = logging.getLogger(__name__)

SEED_VOCAB: dict[str, list[str]] = {
    "food_dining": [
        "restaurant", "uber eats", "doordash", "grubhub", "mcdonald",
        "starbucks", "chipotle", "pizza", "sushi", "burger", "taco",
        "panera", "subway", "wendy", "chick-fil-a", "domino",
    ],
    "groceries": [
        "trader joe", "whole foods", "safeway", "kroger", "aldi",
        "publix", "sprouts", "food lion", "wegmans", "costco",
        "supermarket", "grocery",
    ],
    "subscriptions": [
        "netflix", "spotify", "hulu", "adobe", "chatgpt", "openai",
        "apple one", "amazon prime", "youtube premium", "disney+",
        "paramount", "peacock", "hbo", "dropbox", "microsoft 365",
    ],
    "transportation": [
        "uber", "lyft", "metro", "parking", "gas", "shell", "bp",
        "exxon", "chevron", "transit", "mta", "amtrak", "toll",
    ],
    "shopping": [
        "amazon", "target", "walmart", "ebay", "etsy", "best buy",
        "home depot", "ikea", "zara", "h&m", "nordstrom", "macy",
    ],
    "income": [
        "salary", "payroll", "direct deposit", "venmo credit",
        "zelle received", "refund", "tax return", "dividend",
    ],
    "health": [
        "pharmacy", "gym", "doctor", "cvs", "walgreens", "dentist",
        "hospital", "clinic", "optometrist", "la fitness", "planet fitness",
    ],
    "housing": [
        "rent", "mortgage", "utilities", "electric", "water", "gas utility",
        "internet", "comcast", "verizon", "att", "con ed",
    ],
}


class SemanticCategorizer:
    """Assign categories to transactions using sentence embeddings + HDBSCAN.

    Falls back to keyword matching when the model is not available.
    """

    def __init__(self, cache_dir: Path | None = None, use_embeddings: bool = True):
        if cache_dir is None:
            cache_dir = Path(
                os.environ.get("RECEIPT_MODEL_CACHE", "~/.receipt/models")
            ).expanduser()
        self._cache_dir = cache_dir
        self._use_embeddings = use_embeddings
        self._model = None
        self._seed_embeddings: dict[str, np.ndarray] = {}

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = SentenceTransformer(
                "all-MiniLM-L6-v2", cache_folder=str(self._cache_dir)
            )
            logger.info("Loaded sentence-transformers model.")
        except Exception as exc:
            logger.warning("sentence-transformers unavailable (%s); using keyword fallback.", exc)
            self._model = None
            self._use_embeddings = False

    def _embed(self, texts: list[str]) -> np.ndarray:
        assert self._model is not None  # only called after a successful _load_model
        return self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def _build_seed_embeddings(self) -> None:
        if self._seed_embeddings:
            return
        all_terms: list[str] = []
        all_cats: list[str] = []
        for cat, terms in SEED_VOCAB.items():
            all_terms.extend(terms)
            all_cats.extend([cat] * len(terms))

        vecs = self._embed(all_terms)
        for cat in SEED_VOCAB:
            mask = [c == cat for c in all_cats]
            self._seed_embeddings[cat] = vecs[mask].mean(axis=0)

    def _nearest_category(self, vec: np.ndarray) -> tuple[str, float]:
        best_cat, best_sim = "other", -1.0
        for cat, centroid in self._seed_embeddings.items():
            sim = float(
                np.dot(vec, centroid)
                / (np.linalg.norm(vec) * np.linalg.norm(centroid) + 1e-9)
            )
            if sim > best_sim:
                best_sim = sim
                best_cat = cat
        return best_cat, best_sim

    def _keyword_category(self, description: str) -> tuple[str, float]:
        desc_lower = description.lower()
        for cat, terms in SEED_VOCAB.items():
            for term in terms:
                if term in desc_lower:
                    return cat, 0.9
        return "other", 0.0

    def cache_embeddings(self, path: Path) -> None:
        """Persist seed embeddings to disk."""
        self._load_model()
        if not self._use_embeddings:
            return
        self._build_seed_embeddings()
        data = {k: v.tolist() for k, v in self._seed_embeddings.items()}
        path.write_text(json.dumps(data))

    def load_cached_embeddings(self, path: Path) -> bool:
        if not path.exists():
            return False
        data = json.loads(path.read_text())
        self._seed_embeddings = {k: np.array(v) for k, v in data.items()}
        return True

    def categorize(
        self,
        df: pd.DataFrame,
        audit_log: PipelineAuditLog | None = None,
    ) -> pd.DataFrame:
        """Return df with added columns: category, category_confidence, cluster_id."""
        from receipt.pipeline.audit import AuditLogger

        with AuditLogger(audit_log, "categorize", len(df)) as al:
            result = self._categorize_impl(df)
            al.output_rows = len(result)
            al.metadata["use_embeddings"] = self._use_embeddings
        return result

    def _categorize_impl(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self._use_embeddings:
            self._load_model()

        descriptions = df["description"].tolist()

        if self._use_embeddings and self._model is not None:
            self._build_seed_embeddings()
            vecs = self._embed(descriptions)

            # HDBSCAN clustering
            try:
                import hdbscan

                clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
                cluster_ids = clusterer.fit_predict(vecs).tolist()
            except Exception:
                cluster_ids = [-1] * len(descriptions)

            categories, confidences = zip(
                *[self._nearest_category(v) for v in vecs]
            )
        else:
            cluster_ids = [-1] * len(descriptions)
            result = [self._keyword_category(d) for d in descriptions]
            categories, confidences = zip(*result) if result else ([], [])

        df["category"] = list(categories)
        df["category_confidence"] = list(confidences)
        df["cluster_id"] = cluster_ids
        return df
