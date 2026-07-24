"""Content-quality tests for Narrator.generate_narrative().

Validates that the narrator produces specific, number-grounded narratives free
of generic filler — not just structurally valid JSON.  All API calls are fully
mocked; no real API key is required.

GENERIC_PHRASES is imported directly from receipt.analysis.narrator._LOW_SIGNAL_PHRASES
so the test and the production validator stay in sync automatically.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from receipt.analysis.narrator import _LOW_SIGNAL_PHRASES as GENERIC_PHRASES
from receipt.analysis.narrator import NarrativeReport, Narrator
from receipt.pipeline.aggregator import compute_stats

# ---------------------------------------------------------------------------
# Canonical good-quality narrative used as the mock API response
# ---------------------------------------------------------------------------

_GOOD_NARRATIVE: dict = {
    "tldr": "Chipotle alone consumed 34% of your $547 food budget across 11 visits.",
    "insights": [
        {
            "headline": "Chipotle took 34% of your $547 food budget in 11 visits",
            "detail": (
                "Eleven visits totalling $187 made Chipotle your second-largest single-merchant "
                "spend after rent. At that visit rate it has crossed from occasional treat into a "
                "quasi-fixed cost — the other 9 food merchants combined spent less."
            ),
        },
        {
            "headline": "Netflix at $15.49/month is 11.5% of your $134 subscription total",
            "detail": (
                "At $15.49 recurring, Netflix represents 11.5% of your $134.93 subscription spend. "
                "Combined with 5 other auto-renewing services, you're paying $45/month for content "
                "that exits your account automatically without a conscious purchase decision each time."
            ),
        },
        {
            "headline": "Week of 2026-04-14 spiked to 2.3x your weekly average of $137",
            "detail": (
                "That week totalled $315, driven by 3 transactions at a single electronics retailer. "
                "The remaining 3 weeks averaged $128 — making this a clear outlier, not a new trend."
            ),
        },
    ],
    "next_steps": (
        "Audit the 5 streaming subscriptions: 3 of the $134/month auto-renew without a monthly "
        "prompt. Cancelling 2 mid-tier services saves $29/month or $348/year."
    ),
}

_GOOD_JSON: str = json.dumps(_GOOD_NARRATIVE)

# ---------------------------------------------------------------------------
# Helper: mock API response factory
# ---------------------------------------------------------------------------


def _make_response(json_text: str) -> MagicMock:
    """Return a MagicMock that mimics an Anthropic messages.create() response."""
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text=json_text)]
    return mock_resp


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def narrator() -> Narrator:
    return Narrator(api_key="test-key")


@pytest.fixture
def narrator_stats(sample_df) -> dict:
    """Stats dict derived from the shared sample_df fixture in conftest.py."""
    return compute_stats(sample_df)


# ---------------------------------------------------------------------------
# TestNarratorQuality
# ---------------------------------------------------------------------------


class TestNarratorQuality:
    """Content-quality assertions for Narrator.generate_narrative()."""

    # ------------------------------------------------------------------
    # Internal helper — keeps test bodies clean
    # ------------------------------------------------------------------

    def _run(self, narrator: Narrator, stats: dict, json_text: str) -> NarrativeReport:
        """Call generate_narrative() with a single mocked API response."""
        with patch("receipt.analysis.narrator.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = _make_response(json_text)
            return narrator.generate_narrative(stats, [])

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_tldr_is_concise(self, narrator, narrator_stats):
        report = self._run(narrator, narrator_stats, _GOOD_JSON)
        word_count = len(report.tldr.split())
        assert word_count <= 25, (
            f"tldr has {word_count} words (max 25): {report.tldr!r}"
        )

    def test_insights_contain_numbers(self, narrator, narrator_stats):
        report = self._run(narrator, narrator_stats, _GOOD_JSON)
        has_digit = any(
            any(ch.isdigit() for ch in (i.headline + " " + i.detail))
            for i in report.insights
        )
        assert has_digit, (
            "No insight contains a digit — narrative is entirely generic text with no "
            "data-grounded claims"
        )

    @pytest.mark.parametrize("phrase", GENERIC_PHRASES)
    def test_no_generic_phrases(self, narrator, narrator_stats, phrase):
        report = self._run(narrator, narrator_stats, _GOOD_JSON)
        for insight in report.insights:
            text = (insight.headline + " " + insight.detail).lower()
            assert phrase not in text, (
                f"Generic phrase {phrase!r} found in insight {insight.headline!r}"
            )

    def test_next_steps_is_actionable(self, narrator, narrator_stats):
        report = self._run(narrator, narrator_stats, _GOOD_JSON)
        assert not report.next_steps.startswith("Consider"), (
            f"next_steps opens with 'Consider' (passive hedge): {report.next_steps!r}"
        )
        assert len(report.next_steps) > 30, (
            f"next_steps is too short ({len(report.next_steps)} chars): {report.next_steps!r}"
        )

    def test_insights_have_minimum_count(self, narrator, narrator_stats):
        report = self._run(narrator, narrator_stats, _GOOD_JSON)
        assert len(report.insights) >= 2, (
            f"Only {len(report.insights)} insight(s) returned — minimum is 2"
        )

    def test_insight_detail_is_not_just_headline_rephrased(self, narrator, narrator_stats):
        report = self._run(narrator, narrator_stats, _GOOD_JSON)
        for insight in report.insights:
            assert insight.headline not in insight.detail, (
                f"Headline is a substring of its own detail (rephrasing): {insight.headline!r}"
            )
            assert len(insight.detail) > len(insight.headline), (
                f"Detail ({len(insight.detail)} chars) must be longer than headline "
                f"({len(insight.headline)} chars): {insight.headline!r}"
            )

    def test_malformed_json_triggers_retry(self, narrator, narrator_stats):
        with patch("receipt.analysis.narrator.anthropic.Anthropic") as MockClient:
            mock_create = MockClient.return_value.messages.create
            mock_create.side_effect = [
                _make_response("this is not valid json {{{"),
                _make_response(_GOOD_JSON),
            ]
            report = narrator.generate_narrative(narrator_stats, [])

        assert isinstance(report, NarrativeReport), (
            "generate_narrative() should succeed when the second attempt returns valid JSON"
        )
        assert mock_create.call_count == 2, (
            f"Expected 2 API calls (1 failure + 1 retry), got {mock_create.call_count}"
        )

    def test_all_retries_exhausted_raises_runtime_error(self, narrator, narrator_stats):
        with patch("receipt.analysis.narrator.anthropic.Anthropic") as MockClient:
            mock_create = MockClient.return_value.messages.create
            mock_create.side_effect = [
                _make_response("invalid json attempt 1 {{{"),
                _make_response("invalid json attempt 2 {{{"),
                _make_response("invalid json attempt 3 {{{"),
            ]
            with pytest.raises(RuntimeError, match="Failed to generate narrative"):
                narrator.generate_narrative(narrator_stats, [])
