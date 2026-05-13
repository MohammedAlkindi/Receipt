"""Anthropic API narrative generation."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import anthropic

from receipt.analysis.patterns import Pattern
from receipt.pipeline.drift import DriftReport

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a sharp, warm financial narrator. You have been given structured \
analysis of someone's transaction data including spending stats, detected \
patterns, anomalies, and behavioral drift compared to last period.

Surface 3-5 genuinely interesting observations the person probably has \
not noticed. Write like a smart friend, not a financial advisor. Be \
specific with numbers when surprising. Never moralize. Lead with the \
most interesting insight.

Return only valid JSON with this exact structure:
{
  "insights": [
    {"headline": "bold one-liner", "detail": "2-3 sentence explanation"}
  ],
  "next_steps": "one warm paragraph",
  "tldr": "one sentence summary of the whole month"
}"""


@dataclass
class Insight:
    headline: str
    detail: str


@dataclass
class NarrativeReport:
    tldr: str
    insights: list[Insight]
    next_steps: str
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tldr": self.tldr,
            "insights": [{"headline": i.headline, "detail": i.detail} for i in self.insights],
            "next_steps": self.next_steps,
            "generated_at": self.generated_at,
        }


def _build_prompt(
    stats: dict[str, Any],
    patterns: list[Pattern],
    drift: DriftReport | None,
) -> str:
    parts: list[str] = []

    parts.append("=== SPENDING SUMMARY ===")
    parts.append(f"Total spent: ${abs(stats.get('total_spent', 0)):.2f}")
    parts.append(f"Total income: ${stats.get('total_income', 0):.2f}")
    parts.append(f"Net: ${stats.get('net', 0):.2f}")
    parts.append(f"Transactions analyzed: {stats.get('transaction_count', 0)}")

    if stats.get("subscription_total"):
        parts.append(f"Subscription spend: ${stats['subscription_total']:.2f}")

    if stats.get("largest_single_transaction"):
        lt = stats["largest_single_transaction"]
        parts.append(
            f"Largest transaction: ${abs(lt['amount']):.2f} at {lt['description']}"
        )

    if stats.get("most_frequent_merchant"):
        mf = stats["most_frequent_merchant"]
        parts.append(
            f"Most visited merchant: {mf['merchant']} ({mf['count']} times)"
        )

    parts.append("\n=== SPENDING BY CATEGORY ===")
    for cat, data in stats.get("by_category", {}).items():
        parts.append(f"  {cat}: ${data['total']:.2f} ({data['count']} transactions, avg ${data['avg']:.2f})")

    parts.append("\n=== WEEKLY SPENDING ===")
    for week, total in stats.get("by_week", {}).items():
        parts.append(f"  {week}: ${total:.2f}")

    if patterns:
        parts.append("\n=== DETECTED PATTERNS ===")
        for p in patterns:
            parts.append(f"  [{p.severity.upper()}] {p.type}: {p.headline}")

    if drift:
        parts.append("\n=== BEHAVIORAL DRIFT vs PREVIOUS PERIOD ===")
        if drift.velocity_trend != "stable":
            parts.append(f"  Spending velocity: {drift.velocity_trend}")
        for cat, detail in drift.increased.items():
            parts.append(
                f"  {cat} UP {detail['change_pct']}%: ${detail['previous']} → ${detail['current']}"
            )
        for cat, detail in drift.decreased.items():
            parts.append(
                f"  {cat} DOWN {abs(detail['change_pct'])}%: ${detail['previous']} → ${detail['current']}"
            )
        if drift.new_merchants:
            parts.append(f"  New merchants: {', '.join(drift.new_merchants[:8])}")
        if drift.dropped_merchants:
            parts.append(f"  Dropped merchants: {', '.join(drift.dropped_merchants[:8])}")
        if drift.subscription_drift.get("new"):
            parts.append(f"  New subscriptions: {', '.join(drift.subscription_drift['new'])}")
        if drift.subscription_drift.get("cancelled"):
            parts.append(f"  Cancelled subscriptions: {', '.join(drift.subscription_drift['cancelled'])}")

    return "\n".join(parts)


class Narrator:
    """Generate plain-English financial narratives via the Anthropic API."""

    MODEL = "claude-sonnet-4-20250514"
    MAX_TOKENS = 2000
    MAX_RETRIES = 3
    API_TIMEOUT = 30.0

    def __init__(self, api_key: str):
        self._api_key = api_key

    def generate_narrative(
        self,
        stats: dict[str, Any],
        patterns: list[Pattern],
        drift: DriftReport | None = None,
    ) -> NarrativeReport:
        client = anthropic.Anthropic(api_key=self._api_key)
        base_content = _build_prompt(stats, patterns, drift)

        last_exc: Exception | None = None
        malformed_text: str | None = None

        for attempt in range(self.MAX_RETRIES):
            if malformed_text is not None:
                user_content = (
                    "IMPORTANT: Your previous response was not valid JSON. "
                    "Return ONLY the raw JSON object. No markdown fences, no backticks, "
                    "no explanation before or after the JSON.\n\n"
                    f"Your previous (invalid) response was:\n{malformed_text}\n\n"
                    f"Now generate the correct JSON for this data:\n\n{base_content}"
                )
            else:
                user_content = base_content

            raw_text: str = ""
            try:
                response = client.messages.create(
                    model=self.MODEL,
                    max_tokens=self.MAX_TOKENS,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_content}],
                    timeout=self.API_TIMEOUT,
                )
                raw_text = response.content[0].text
                data = json.loads(raw_text)
                insights = [
                    Insight(headline=i["headline"], detail=i["detail"])
                    for i in data.get("insights", [])
                ]
                return NarrativeReport(
                    tldr=data.get("tldr", ""),
                    insights=insights,
                    next_steps=data.get("next_steps", ""),
                )
            except json.JSONDecodeError as exc:
                logger.debug("Malformed JSON response (attempt %d): %s", attempt + 1, raw_text)
                logger.warning("JSON parse error on attempt %d: %s", attempt + 1, exc)
                malformed_text = raw_text
                last_exc = exc
            except Exception as exc:
                logger.warning("API error on attempt %d: %s", attempt + 1, exc)
                last_exc = exc
                malformed_text = None
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)

        raise RuntimeError(
            f"Failed to generate narrative after {self.MAX_RETRIES} attempts: {last_exc}"
        )
