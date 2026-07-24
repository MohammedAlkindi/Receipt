"""Anthropic API narrative generation.

ASSUMPTIONS:
- Stats dict fields are exactly what compute_stats() returns in aggregator.py.
  No field is referenced here that isn't present in that function's return dict.
- Pattern.severity is constrained to ("critical", "warning", "info") per patterns.py.
- DriftReport attributes (velocity_trend, increased, decreased, new_merchants,
  dropped_merchants, subscription_drift) are accessed identically to the original.
- temperature=0.7 balances specificity with creative recombination; lower it (e.g.
  0.3) if output grows too unpredictable for production use.
- by_merchant entries have the shape {description: {total: float, count: int}}
  as produced by aggregator.compute_stats() via DataFrame.to_dict(orient="index").
"""

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

_SEVERITY_ORDER: dict[str, int] = {"critical": 0, "warning": 1, "info": 2}

_LOW_SIGNAL_PHRASES: list[str] = [
    "you spent",
    "consider reducing",
    "it looks like",
    "it's worth noting",
    "it is worth noting",
    "you may want to",
    "you might want to",
    "based on your spending",
    "overall, your",
    "in conclusion",
    "as you can see",
    "interestingly,",
    "as a reminder",
    "you should consider",
]

_SYSTEM_PROMPT = """\
## ROLE

You are a forensic spending analyst embedded inside a personal finance app. \
Your job is to read structured transaction data and surface things the user \
almost certainly has not noticed. You are not a budgeting coach, not a life \
advisor, and not a generic finance writer. You write the way a sharp analyst \
friend would talk after staring at someone's bank statement for ten minutes: \
specific, direct, occasionally surprising, never preachy. \
You reason step-by-step through the data before committing to any claim.

## CHAIN-OF-THOUGHT INSTRUCTION

Before writing each insight, silently answer these questions:
1. What number in this data is genuinely surprising relative to the other numbers?
2. Is there a ratio or comparison that reframes a flat figure in a non-obvious way?
3. Is there a time-based trend (week spike, late-night pattern, day-of-week) \
with a concrete dollar consequence?
4. Is there merchant concentration or category dominance the user probably \
hasn't clocked?
5. Would a smart person glancing at this data already know this insight, or \
would it stop them cold?

Discard any insight where the answer to question 5 is "they'd already know this." \
Only write insights that pass.

## WHAT MAKES A GOOD INSIGHT

- Anchored in a specific dollar amount, ratio, or count from the data
- Names the exact merchant, category, or calendar week driving the claim
- Reframes a familiar number in a way that changes how the user sees it
- Leads with the most interesting fact — no wind-up, no preamble
- Treats the user as an intelligent adult who can handle unvarnished truth
- Could not have been written without this specific dataset

## WHAT TO AVOID

Never use these phrases or patterns:
- "You spent" as a sentence opener — flat, adds no value
- "Consider reducing" / "You should consider" — moralizing
- "It looks like" — hedge that weakens the claim
- "It's worth noting" / "It is worth noting" — pure filler
- "You may want to" / "You might want to" — unsolicited advice
- "Based on your spending" — unnecessary preamble
- "Interestingly," as a sentence opener — tells rather than shows
- "Overall," / "In conclusion," / "As you can see," — filler transitions
- "As a reminder," — condescending
- Any sentence that could appear verbatim in a generic personal finance article

Do not moralize about spending categories. \
Do not give advice that isn't tied to a specific number in this data. \
Do not write anything that would be equally true for any person.

## EXAMPLE: GOOD vs. BAD INSIGHT

BAD INSIGHT
  headline: "You spent a lot on food this month"
  detail: "It looks like your food spending was higher than usual. \
Consider reducing dining out to save money. You might want to cook more at home."
  WHY BAD: Flat opener. Moralizes twice. No specific number, merchant, or \
comparison. Could appear in any personal finance newsletter.

GOOD INSIGHT
  headline: "Chipotle alone consumed 34% of your entire food budget"
  detail: "Eleven visits for $187 total makes Chipotle your second-largest \
single-merchant spend after rent. At that frequency it has crossed from \
occasional treat into a quasi-fixed cost. The other nine food merchants \
combined cost less."
  WHY GOOD: Specific merchant, exact dollar amount, exact percentage, \
surprising comparison (vs. rent), reframes the behavior without moralizing.

## OUTPUT FORMAT

Return ONLY valid JSON. No markdown fences, no backticks, no text before or after.

Schema:
{
  "insights": [
    {
      "headline": "string — bold one-liner, max 12 words, anchored in a specific number",
      "detail": "string — 2-3 sentences, specific figures, no moralizing"
    }
  ],
  "next_steps": "string — one specific paragraph, 2-4 sentences, grounded in a pattern from this data",
  "tldr": "string — one sentence, max 25 words, the single most surprising fact about this period"
}

Constraints:
- insights: 3 to 5 items, sorted most-surprising first
- Prioritize patterns marked CRITICAL or WARNING when choosing what to highlight
- Output must parse as valid JSON with no trailing commas
- Do not wrap the JSON in any other text"""


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
            "schema_version": 1,
            "tldr": self.tldr,
            "insights": [{"headline": i.headline, "detail": i.detail} for i in self.insights],
            "next_steps": self.next_steps,
            "generated_at": self.generated_at,
        }


def _build_prompt(
    stats: dict[str, Any],
    patterns: list[Pattern],
    drift: DriftReport | None,
    anomalies: list[dict[str, Any]] | None = None,
) -> str:
    parts: list[str] = []

    total_spent = abs(stats.get("total_spent", 0))
    total_income = stats.get("total_income", 0)
    net = stats.get("net", 0)
    tx_count = stats.get("transaction_count", 0)
    expense_count = stats.get("expense_count", 0)
    income_count = stats.get("income_count", 0)
    by_week = stats.get("by_week", {})
    cat_count = len(stats.get("by_category", {}))

    # Determine date range from week keys if available
    date_range = ""
    if by_week:
        sorted_weeks = sorted(by_week.keys())
        date_range = f" from {sorted_weeks[0]} to {sorted_weeks[-1]}"

    # -------------------------------------------------------------------------
    parts.append("## CONTEXT")
    parts.append(
        f"This dataset covers {tx_count} transactions "
        f"({expense_count} expenses, {income_count} income events) "
        f"spanning {len(by_week)} calendar week(s){date_range} "
        f"across {cat_count} spending categories. "
        "The user wants to understand where their money actually went "
        "and what non-obvious patterns exist in their behavior."
    )
    parts.append("")

    # -------------------------------------------------------------------------
    parts.append("## SPENDING OVERVIEW")
    parts.append(f"- Total spent: ${total_spent:.2f}")
    parts.append(f"- Total income: ${total_income:.2f}")
    net_sign = "+" if net >= 0 else ""
    parts.append(f"- Net cash flow: {net_sign}${net:.2f}")
    if stats.get("subscription_total"):
        sub_total = stats["subscription_total"]
        sub_pct = (sub_total / total_spent * 100) if total_spent else 0.0
        parts.append(
            f"- Subscription spend: ${sub_total:.2f} ({sub_pct:.1f}% of total outflow)"
        )
    if stats.get("largest_single_transaction"):
        lt = stats["largest_single_transaction"]
        parts.append(
            f"- Largest single transaction: ${abs(lt['amount']):.2f} "
            f"at {lt['description']} on {lt['date'][:10]}"
        )
    if stats.get("most_frequent_merchant"):
        mf = stats["most_frequent_merchant"]
        parts.append(
            f"- Most visited merchant: {mf['merchant']} ({mf['count']} visits)"
        )
    parts.append("")

    # -------------------------------------------------------------------------
    parts.append("## CATEGORY BREAKDOWN")
    by_cat = stats.get("by_category", {})
    if by_cat:
        sorted_cats = sorted(by_cat.items(), key=lambda kv: kv[1]["total"], reverse=True)
        for cat, data in sorted_cats:
            pct = (data["total"] / total_spent * 100) if total_spent else 0.0
            parts.append(
                f"- {cat}: ${data['total']:.2f} ({pct:.1f}% of spend) "
                f"— {data['count']} txns, avg ${data['avg']:.2f}"
            )
    else:
        parts.append("- No category data available")
    parts.append("")

    # -------------------------------------------------------------------------
    by_merchant = stats.get("by_merchant", {})
    if by_merchant:
        parts.append("## TOP MERCHANTS (by total spend)")
        for merchant, mdata in list(by_merchant.items())[:10]:
            m_pct = (mdata["total"] / total_spent * 100) if total_spent else 0.0
            parts.append(
                f"- {merchant}: ${mdata['total']:.2f} ({m_pct:.1f}% of spend) "
                f"— {mdata['count']} visits"
            )
        parts.append("")

    # -------------------------------------------------------------------------
    parts.append("## WEEKLY PATTERN")
    if by_week:
        weekly_vals = list(by_week.values())
        avg_weekly = sum(weekly_vals) / len(weekly_vals)
        parts.append(f"- Weekly average: ${avg_weekly:.2f}")
        for week, total in sorted(by_week.items()):
            annotation = ""
            if avg_weekly > 0:
                ratio = total / avg_weekly
                if ratio >= 1.5:
                    annotation = f"  *** SPIKE: {ratio:.1f}x the weekly average"
                elif ratio <= 0.5:
                    annotation = f"  (low week: {ratio:.1f}x avg)"
            parts.append(f"- {week}: ${total:.2f}{annotation}")
    else:
        parts.append("- No weekly data available")
    parts.append("")

    # -------------------------------------------------------------------------
    if patterns:
        sorted_patterns = sorted(
            patterns, key=lambda p: _SEVERITY_ORDER.get(p.severity, 99)
        )
        parts.append("## DETECTED PATTERNS")
        for p in sorted_patterns:
            parts.append(f"- [{p.severity.upper()}] {p.type}: {p.headline}")
        parts.append("")

    # -------------------------------------------------------------------------
    if anomalies:
        parts.append("## STATISTICAL OUTLIERS (Isolation Forest)")
        for a in anomalies:
            reason = f" — {a['reason']}" if a.get("reason") else ""
            parts.append(
                f"- ${abs(a['amount']):.2f} at {a['description']} on {a['date']}{reason}"
            )
        parts.append("")

    # -------------------------------------------------------------------------
    if drift:
        parts.append("## BEHAVIORAL DRIFT vs PREVIOUS PERIOD")
        if drift.velocity_trend != "stable":
            parts.append(f"- Spending velocity: {drift.velocity_trend}")
        for cat, detail in drift.increased.items():
            if detail.get("change_pct") is None:
                parts.append(f"- {cat} NEW this period: ${detail['current']}")
            else:
                parts.append(
                    f"- {cat} UP {detail['change_pct']}%: "
                    f"${detail['previous']} → ${detail['current']}"
                )
        for cat, detail in drift.decreased.items():
            parts.append(
                f"- {cat} DOWN {abs(detail['change_pct'])}%: "
                f"${detail['previous']} → ${detail['current']}"
            )
        if drift.new_merchants:
            parts.append(
                f"- New merchants this period: {', '.join(drift.new_merchants[:8])}"
            )
        if drift.dropped_merchants:
            parts.append(
                f"- Merchants no longer appearing: {', '.join(drift.dropped_merchants[:8])}"
            )
        if drift.subscription_drift.get("new"):
            parts.append(
                f"- New subscriptions: {', '.join(drift.subscription_drift['new'])}"
            )
        if drift.subscription_drift.get("cancelled"):
            parts.append(
                f"- Cancelled subscriptions: {', '.join(drift.subscription_drift['cancelled'])}"
            )
        parts.append("")

    # -------------------------------------------------------------------------
    parts.append("## YOUR TASK")
    parts.append(
        "Using only the data above, produce 3–5 insights sorted by how surprising "
        "they are (most surprising first). Reason through what is non-obvious before "
        "writing. Prioritize CRITICAL and WARNING patterns. Name specific merchants, "
        "exact dollar amounts, and precise ratios. Discard any insight a generic "
        "finance article could have written without this data. "
        "Return only valid JSON matching the schema in your system prompt."
    )

    return "\n".join(parts)


class Narrator:
    """Generate plain-English financial narratives via the Anthropic API."""

    MODEL = "claude-sonnet-4-20250514"
    MAX_TOKENS = 2000
    MAX_RETRIES = 3
    API_TIMEOUT = 30.0

    def __init__(self, api_key: str, temperature: float = 0.7):
        self._api_key = api_key
        self._temperature = temperature

    @staticmethod
    def _apply_quality_guard(
        insights: list[Insight],
    ) -> tuple[list[Insight], list[Insight]]:
        """Split insights into (kept, suppressed) by low-signal phrase content.

        An insight is suppressed when its text contains any generic filler
        phrase from _LOW_SIGNAL_PHRASES.
        """
        kept: list[Insight] = []
        suppressed: list[Insight] = []
        for insight in insights:
            text = (insight.headline + " " + insight.detail).lower()
            if any(phrase in text for phrase in _LOW_SIGNAL_PHRASES):
                suppressed.append(insight)
            else:
                kept.append(insight)
        return kept, suppressed

    def generate_narrative(
        self,
        stats: dict[str, Any],
        patterns: list[Pattern],
        drift: DriftReport | None = None,
        audit_log: Any | None = None,
        anomalies: list[dict[str, Any]] | None = None,
    ) -> NarrativeReport:
        from receipt.pipeline.audit import AuditLogger

        tx_count = stats.get("transaction_count", 0)
        with AuditLogger(audit_log, "generate_narrative", tx_count) as al:
            report = self._generate_narrative_impl(stats, patterns, drift, al, anomalies)
            al.output_rows = 0
        return report

    def _generate_narrative_impl(
        self,
        stats: dict[str, Any],
        patterns: list[Pattern],
        drift: DriftReport | None,
        al: Any,
        anomalies: list[dict[str, Any]] | None = None,
    ) -> NarrativeReport:
        client = anthropic.Anthropic(api_key=self._api_key)
        base_content = _build_prompt(stats, patterns, drift, anomalies)

        last_exc: Exception | None = None
        retry_reason: str | None = None  # "malformed" | "low_quality"
        malformed_text: str | None = None
        fallback_report: NarrativeReport | None = None

        for attempt in range(self.MAX_RETRIES):
            if retry_reason == "malformed":
                user_content = (
                    "IMPORTANT: Your previous response was not valid JSON. "
                    "Return ONLY the raw JSON object. No markdown fences, no backticks, "
                    "no explanation before or after the JSON.\n\n"
                    f"Your previous (invalid) response was:\n{malformed_text}\n\n"
                    f"Now generate the correct JSON for this data:\n\n{base_content}"
                )
            elif retry_reason == "low_quality":
                user_content = (
                    "IMPORTANT: Your previous insights were too generic or empty. "
                    "Every insight must cite a specific dollar amount, merchant, or "
                    "ratio from the data and must not use filler phrases like "
                    "'you spent', 'consider reducing', or 'it looks like'. "
                    f"Regenerate the JSON for this data:\n\n{base_content}"
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
                    temperature=self._temperature,
                )
                raw_text = response.content[0].text
                data = json.loads(raw_text)
                insights = [
                    Insight(headline=i["headline"], detail=i["detail"])
                    for i in data.get("insights", [])
                ]
                tldr = data.get("tldr", "")
                next_steps = data.get("next_steps", "")

                kept, suppressed = self._apply_quality_guard(insights)
                if suppressed:
                    logger.info(
                        "Quality guard suppressed %d/%d low-signal insight(s).",
                        len(suppressed),
                        len(insights),
                    )

                if kept:
                    al.metadata["api_attempts"] = attempt + 1
                    al.metadata["insights_suppressed"] = len(suppressed)
                    return NarrativeReport(tldr=tldr, insights=kept, next_steps=next_steps)

                # Nothing usable survived the guard (empty or all-generic).
                # Remember the first parseable response as a last resort so we
                # never return a completely empty narrative, then retry.
                if insights and fallback_report is None:
                    fallback_report = NarrativeReport(
                        tldr=tldr, insights=insights, next_steps=next_steps
                    )
                retry_reason = "low_quality"
                last_exc = RuntimeError("all insights failed the quality guard")
            except json.JSONDecodeError as exc:
                logger.debug("Malformed JSON response (attempt %d): %s", attempt + 1, raw_text)
                logger.warning("JSON parse error on attempt %d: %s", attempt + 1, exc)
                retry_reason = "malformed"
                malformed_text = raw_text
                last_exc = exc
            except Exception as exc:
                logger.warning("API error on attempt %d: %s", attempt + 1, exc)
                last_exc = exc
                retry_reason = None
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)

        al.metadata["api_attempts"] = self.MAX_RETRIES
        if fallback_report is not None:
            logger.warning(
                "Quality guard: all insights were low-signal after %d attempts; "
                "returning the best-effort narrative rather than an empty one.",
                self.MAX_RETRIES,
            )
            return fallback_report
        raise RuntimeError(
            f"Failed to generate narrative after {self.MAX_RETRIES} attempts: {last_exc}"
        )
