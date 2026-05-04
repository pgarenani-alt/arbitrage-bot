"""
AI-powered market matching — Component B of the course project.

Uses Claude to:
  1. Semantically match Kalshi markets to Polymarket markets
     (the platforms describe the same event with different wording)
  2. Generate a plain-English trade recommendation for each matched pair

Claude model used: claude-haiku-4-5-20251001 (fast, cheap for batch matching)
                   claude-sonnet-4-6 for the richer trade explanation
"""
import os
import json
import anthropic
from typing import Optional

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# 1. Batch market matching
# ─────────────────────────────────────────────────────────────────────────────

_MATCH_SYSTEM = """\
You are an expert at matching prediction market questions across platforms.
Kalshi and Polymarket often list the same real-world event with different
question wording, date formats, or phrasing.

Given a Kalshi question and a list of Polymarket candidates, respond ONLY with
valid JSON — no markdown fences, no commentary.

Response format:
{
  "best_match_index": <integer or null>,
  "confidence": <"high"|"medium"|"low">,
  "reasoning": "<one sentence>"
}

Rules:
- Return the index of the single best Polymarket candidate (0-based).
- If no candidate is about the same event, return null for best_match_index.
- Confidence "high" means both questions clearly resolve on the same event/date.
- Confidence "medium" means likely the same event but with some ambiguity.
- Confidence "low" means a possible match but uncertain.
"""


def match_markets(
    kalshi_question: str,
    poly_candidates: list[str],
    max_candidates: int = 10,
) -> dict:
    """
    Ask Claude Haiku to find the Polymarket market that matches a Kalshi market.

    Returns:
        {
            "best_match_index": int | None,
            "confidence": "high" | "medium" | "low",
            "reasoning": str
        }
    """
    if not poly_candidates:
        return {"best_match_index": None, "confidence": "low", "reasoning": "No candidates."}

    candidates_text = "\n".join(
        f"  [{i}] {q}" for i, q in enumerate(poly_candidates[:max_candidates])
    )
    user_msg = (
        f"Kalshi question:\n  {kalshi_question}\n\n"
        f"Polymarket candidates:\n{candidates_text}"
    )

    try:
        client = _get_client()
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=_MATCH_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = resp.content[0].text.strip()
        return json.loads(raw)
    except Exception as e:
        return {
            "best_match_index": None,
            "confidence": "low",
            "reasoning": f"Matching unavailable: {e}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Trade recommendation generation
# ─────────────────────────────────────────────────────────────────────────────

_RECOMMEND_SYSTEM = """\
You are a quantitative analyst specialising in prediction market arbitrage.
Given data about a cross-platform arbitrage opportunity, write a concise but
insightful trade recommendation.

Include:
- What to buy on each platform and at what price
- Expected profit and why the opportunity exists
- Key risks (counterparty, liquidity, time decay, fees)
- One-sentence event context: what is this market about?

Write 3–5 short paragraphs. No bullet-point lists. Be direct and analytical.
"""


def generate_recommendation(opportunity: dict) -> str:
    """
    Generate an AI trade recommendation for a matched arbitrage opportunity.

    opportunity dict keys:
        kalshi_question, poly_question, kalshi_yes_price, poly_yes_price,
        kalshi_platform_action, poly_platform_action,
        combined_cost, expected_profit_pct, ml_confidence, category,
        days_to_resolution
    """
    user_msg = f"""
Arbitrage opportunity details:

Kalshi market : "{opportunity['kalshi_question']}"
Polymarket    : "{opportunity['poly_question']}"
Category      : {opportunity['category']}
Days to resolve: {opportunity['days_to_resolution']:.0f}

Prices:
  Kalshi  YES = {opportunity['kalshi_yes_price']:.3f}  |  NO = {1-opportunity['kalshi_yes_price']:.3f}
  Polymarket YES = {opportunity['poly_yes_price']:.3f}  |  NO = {1-opportunity['poly_yes_price']:.3f}

Trade:
  Buy {opportunity['kalshi_platform_action']} on Kalshi  @ {opportunity['kalshi_trade_price']:.3f}
  Buy {opportunity['poly_platform_action']} on Polymarket @ {opportunity['poly_trade_price']:.3f}
  Combined cost = {opportunity['combined_cost']:.3f}
  Expected profit = {opportunity['expected_profit_pct']:.1f}% (before fees)

ML model confidence that this is a genuine, fee-profitable opportunity: {opportunity['ml_confidence']:.0%}
"""

    try:
        client = _get_client()
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=_RECOMMEND_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        return (
            f"AI analysis unavailable ({e}).\n\n"
            f"Summary: Buy {'YES' if opportunity['kalshi_yes_price'] < 0.5 else 'NO'} on Kalshi "
            f"and {'YES' if opportunity['poly_yes_price'] < 0.5 else 'NO'} on Polymarket for a "
            f"{opportunity['expected_profit_pct']:.1f}% expected return."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Bulk matching helper (used by arbitrage_engine)
# ─────────────────────────────────────────────────────────────────────────────

def bulk_match(kalshi_markets: list[dict], poly_markets: list[dict]) -> list[tuple[dict, dict, dict]]:
    """
    For each Kalshi market, find the best Polymarket partner.
    Returns list of (kalshi_market, poly_market, match_meta) triples
    where confidence is "high" or "medium".
    """
    poly_questions = [m["question"] for m in poly_markets]
    matched = []

    for km in kalshi_markets:
        result = match_markets(km["question"], poly_questions)
        idx = result.get("best_match_index")
        conf = result.get("confidence", "low")

        if idx is None or conf == "low":
            continue
        if idx >= len(poly_markets):
            continue

        matched.append((km, poly_markets[idx], result))

    return matched
