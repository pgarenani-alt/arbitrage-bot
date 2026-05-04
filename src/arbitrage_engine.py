"""
Arbitrage detection engine — combines ML scoring with live market data.

Steps:
  1. Fetch active markets from both platforms
  2. Use Claude to match markets across platforms (or fall back to keyword heuristic)
  3. Compute arbitrage metrics for each matched pair
  4. Score each opportunity with the trained ML model
  5. Return a ranked list of opportunities
"""
import os
import re
import math
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Optional

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Fee model
KALSHI_FEE     = 0.07   # 7% of winnings (on winning trade)
POLY_FEE_ROUND = 0.02   # ~2% round-trip on Polymarket
MIN_PROFIT_PCT = 1.0    # only surface opportunities with >1% expected profit


# ─────────────────────────────────────────────────────────────────────────────
# ML model loader
# ─────────────────────────────────────────────────────────────────────────────

_model         = None
_feature_names = None
_category_map  = None


def _load_model():
    global _model, _feature_names, _category_map
    model_path = os.path.join(MODEL_DIR, "arbitrage_scorer.pkl")
    feat_path  = os.path.join(MODEL_DIR, "feature_names.pkl")
    cat_path   = os.path.join(MODEL_DIR, "category_map.pkl")
    if os.path.exists(model_path) and os.path.exists(feat_path):
        _model         = joblib.load(model_path)
        _feature_names = joblib.load(feat_path)
        _category_map  = joblib.load(cat_path) if os.path.exists(cat_path) else {}
    else:
        _model = None


_load_model()


def _ml_score(opp: dict) -> float:
    """
    Returns the ML model's confidence that this is a genuine arbitrage.

    The model is a calibration estimator: it predicts P(YES resolves) using
    market features.  We score the opportunity by measuring how far the model's
    predicted probability deviates from both platforms' prices — a larger
    deviation means one platform is morelikely mispriced.
    Score = clip(|model_prob - kalshi_price| + |model_prob - poly_price| - 0.02, 0, 0.8)
    normalised to [0,1].
    Falls back to a heuristic when no model is loaded.
    """
    if _model is None:
        return _heuristic_score(opp)

    cat_map = _category_map or {}
    cat_enc = cat_map.get(opp.get("category", "other").lower(), 7)

    yes_price = opp["kalshi_yes_price"]
    vol       = opp.get("min_volume_usd", 0)
    log_vol   = math.log1p(vol)
    days      = opp.get("days_to_resolution", 14)

    # Features match FEATURE_COLS in train_model.py:
    # yes_price, log_volume, days_to_resolution, category_enc,
    # price_confidence, extreme_price, log_vol_x_price
    price_conf  = log_vol * (1 - 2 * abs(yes_price - 0.5))
    extreme     = int(yes_price < 0.10 or yes_price > 0.90)
    log_vol_x_p = log_vol * yes_price

    x = np.array([[yes_price, log_vol, float(days), float(cat_enc),
                   price_conf, float(extreme), log_vol_x_p]])
    try:
        model_prob = float(_model.predict_proba(x)[0][1])
        # How much does the model disagree with each platform?
        kalshi_diff = abs(model_prob - opp["kalshi_yes_price"])
        poly_diff   = abs(model_prob - opp["poly_yes_price"])
        # High disagreement → opportunity is more likely real (one platform is wrong)
        raw_conf = min((kalshi_diff + poly_diff - 0.02) * 3.5, 0.95)
        return max(raw_conf, 0.05)
    except Exception:
        return _heuristic_score(opp)


def _heuristic_score(opp: dict) -> float:
    """Simple rule-based confidence when no model is available."""
    profit = opp.get("expected_profit_pct", 0) / 100
    vol    = opp.get("min_volume_usd", 0)
    days   = opp.get("days_to_resolution", 30)

    score = min(profit * 10, 0.5)   # bigger spread → higher score
    if vol > 50_000:
        score += 0.1
    if vol > 200_000:
        score += 0.1
    if 5 < days < 60:
        score += 0.1
    return min(score, 0.95)


# ─────────────────────────────────────────────────────────────────────────────
# Keyword-based fallback matcher (no Claude needed)
# ─────────────────────────────────────────────────────────────────────────────

def _keyword_match(km: dict, poly_markets: list[dict]) -> Optional[dict]:
    """
    Fuzzy keyword overlap between Kalshi and Polymarket question strings.
    Used as a fallback when Claude is unavailable or for pre-filtering.
    """
    k_words = set(_tokenise(km["question"]))
    best_score = 0
    best_pm    = None

    for pm in poly_markets:
        p_words = set(_tokenise(pm["question"]))
        if not p_words:
            continue
        overlap = len(k_words & p_words) / max(len(k_words | p_words), 1)
        if overlap > best_score:
            best_score = overlap
            best_pm = pm

    if best_score >= 0.25:
        return best_pm
    return None


def _tokenise(text: str) -> list[str]:
    stop = {"will", "the", "a", "an", "of", "in", "be", "to", "by", "at", "or",
            "is", "are", "was", "for", "on", "and", "with", "its", "it"}
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in stop and len(t) > 1]


# ─────────────────────────────────────────────────────────────────────────────
# Core arbitrage computation
# ─────────────────────────────────────────────────────────────────────────────

def _days_until(date_str: str) -> float:
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S+00:00", "%Y-%m-%d"):
        try:
            end = datetime.strptime(date_str[:len(fmt)+2].rstrip("Z"), fmt.rstrip("Z"))
            return max(0.0, (end - datetime.utcnow()).total_seconds() / 86_400)
        except Exception:
            continue
    return 30.0


def _compute_opportunity(km: dict, pm: dict, match_meta: dict) -> Optional[dict]:
    """
    Given a matched pair, compute the arbitrage trade structure.

    Arbitrage logic:
      If Kalshi YES < Polymarket YES:
        → Buy YES on Kalshi, Buy NO on Polymarket
        Combined cost = kalshi_yes + poly_no
        Payout = $1 if YES resolves; Payout = $1 if NO resolves
        Either way payout = $1, so profit = 1 - combined_cost

      If Polymarket YES < Kalshi YES:
        → Buy YES on Polymarket, Buy NO on Kalshi
    """
    k_yes = km["yes_price"]
    p_yes = pm["yes_price"]

    if k_yes < p_yes:
        # Buy YES on Kalshi, NO on Polymarket
        trade_a_platform = "kalshi"
        trade_a_side     = "YES"
        trade_a_price    = k_yes
        trade_b_platform = "polymarket"
        trade_b_side     = "NO"
        trade_b_price    = 1.0 - p_yes
    else:
        # Buy YES on Polymarket, NO on Kalshi
        trade_a_platform = "polymarket"
        trade_a_side     = "YES"
        trade_a_price    = p_yes
        trade_b_platform = "kalshi"
        trade_b_side     = "NO"
        trade_b_price    = 1.0 - k_yes

    combined_cost = round(trade_a_price + trade_b_price, 4)
    gross_profit  = round(1.0 - combined_cost, 4)

    # Fee adjustment: Kalshi takes 7% of winnings; Polymarket ~2% round-trip
    fee_cost           = round(KALSHI_FEE * 0.5 + POLY_FEE_ROUND, 4)
    net_profit_pct     = round((gross_profit - fee_cost) * 100, 2)
    expected_profit_pct = round(gross_profit * 100, 2)   # gross (before fees), shown in UI

    if gross_profit <= 0:
        return None   # no arbitrage

    days = _days_until(km.get("end_date", "") or pm.get("end_date", ""))
    category = km.get("category") or pm.get("category") or "other"
    min_vol  = min(km.get("volume_usd", 0), pm.get("volume_usd", 0))

    opp = {
        # Identity
        "kalshi_id":       km["id"],
        "poly_id":         pm["id"],
        "kalshi_question": km["question"],
        "poly_question":   pm["question"],
        "category":        category,

        # Prices
        "kalshi_yes_price": k_yes,
        "poly_yes_price":   p_yes,
        "price_spread":     round(abs(k_yes - p_yes), 4),

        # Trade instructions
        # trade_a is always the YES buyer; trade_b is always the NO buyer.
        # kalshi/poly actions and prices derive from which platform is trade_a.
        "kalshi_platform_action": f"Buy {'YES' if trade_a_platform == 'kalshi' else 'NO'}",
        "poly_platform_action":   f"Buy {'YES' if trade_a_platform == 'polymarket' else 'NO'}",
        "kalshi_trade_price": k_yes if trade_a_platform == "kalshi" else (1.0 - k_yes),
        "poly_trade_price":   p_yes if trade_a_platform == "polymarket" else (1.0 - p_yes),
        "buy_price_a":       trade_a_price,
        "buy_price_b":       trade_b_price,

        # Economics
        "combined_cost":        combined_cost,
        "gross_profit":         gross_profit,
        "expected_profit_pct":  expected_profit_pct,
        "net_profit_pct":       net_profit_pct,

        # Context
        "days_to_resolution": round(days, 1),
        "min_volume_usd":     min_vol,
        "kalshi_url":         km.get("url", ""),
        "poly_url":           pm.get("url", ""),
        "match_confidence":   match_meta.get("confidence", "medium"),
        "match_reasoning":    match_meta.get("reasoning", ""),
    }

    # ML score
    opp["ml_confidence"] = _ml_score(opp)

    return opp


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def find_opportunities(
    kalshi_markets:  list[dict],
    poly_markets:    list[dict],
    use_ai_matching: bool = True,
    min_profit_pct:  float = MIN_PROFIT_PCT,
) -> list[dict]:
    """
    Main entry point.  Returns a sorted list of arbitrage opportunities.
    """
    opportunities = []
    used_poly_ids = set()

    if use_ai_matching:
        try:
            from src.market_matcher import bulk_match
            matched_triples = bulk_match(kalshi_markets, poly_markets)
            for km, pm, meta in matched_triples:
                opp = _compute_opportunity(km, pm, meta)
                if opp and opp["expected_profit_pct"] >= min_profit_pct:
                    opportunities.append(opp)
                    used_poly_ids.add(pm["id"])
        except Exception:
            pass   # fall through to keyword matcher

    # Keyword fallback for any unmatched Kalshi markets
    for km in kalshi_markets:
        if any(o["kalshi_id"] == km["id"] for o in opportunities):
            continue
        unmatched_poly = [p for p in poly_markets if p["id"] not in used_poly_ids]
        pm = _keyword_match(km, unmatched_poly)
        if pm:
            opp = _compute_opportunity(km, pm, {"confidence": "medium", "reasoning": "keyword overlap"})
            if opp and opp["expected_profit_pct"] >= min_profit_pct:
                opportunities.append(opp)
                used_poly_ids.add(pm["id"])

    # Sort by ML confidence × expected profit
    opportunities.sort(
        key=lambda o: o["ml_confidence"] * o["expected_profit_pct"],
        reverse=True,
    )
    return opportunities


def opportunities_to_dataframe(opportunities: list[dict]) -> pd.DataFrame:
    if not opportunities:
        return pd.DataFrame()

    rows = []
    for o in opportunities:
        rows.append({
            "Event":            _truncate(o["kalshi_question"], 55),
            "Category":         o["category"].title(),
            "Kalshi Price":     f"{o['kalshi_yes_price']:.2f}",
            "Poly Price":       f"{o['poly_yes_price']:.2f}",
            "Spread":           f"{o['price_spread']:.3f}",
            "Gross Profit %":   f"{o['expected_profit_pct']:.2f}%",
            "ML Confidence":    f"{o['ml_confidence']:.0%}",
            "Days Left":        f"{o['days_to_resolution']:.0f}",
            "Match Quality":    o["match_confidence"].title(),
            "_raw":             o,
        })
    return pd.DataFrame(rows)


def _truncate(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[:max_len - 1] + "…"
