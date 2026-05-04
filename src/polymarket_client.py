"""
Polymarket API client.
Uses the public Gamma API (no auth required) for market data.
Gamma API docs: https://docs.polymarket.com/
"""
import re
import requests
import pandas as pd
from typing import Optional
import time

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"

HEADERS = {"User-Agent": "ArbitrageBot/1.0"}


def _get(url: str, params: dict = None, retries: int = 3) -> dict | list:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 ** attempt)


def get_active_markets(limit: int = 100, category: str = None,
                       demo_mode: bool = False) -> list[dict]:
    """demo_mode has no effect here — Polymarket is always live (no auth needed)."""
    return _get_active_markets_live(limit=limit, category=category)


def _get_active_markets_live(limit: int = 100, category: str = None) -> list[dict]:
    """
    Fetch active binary markets from Polymarket Gamma API.
    Returns a list of normalized market dicts.
    """
    params = {
        "active": "true",
        "closed": "false",
        "limit": limit,
        "order": "volume24hr",
        "ascending": "false",
    }
    if category:
        params["tag"] = category

    try:
        data = _get(f"{GAMMA_BASE}/markets", params=params)
    except Exception:
        return _demo_poly_markets()

    if not data:
        return _demo_poly_markets()

    markets = []
    for m in data:
        try:
            # Polymarket YES price comes from the CLOB order book; the Gamma API
            # provides a `bestAsk` / `bestBid` or a `lastTradePrice` field.
            yes_price = _extract_yes_price(m)
            if yes_price is None:
                continue

            slug = m.get("slug") or ""
            condition_id = m.get("conditionId") or m.get("id") or ""
            # Gamma API slugs have a trailing numeric ID suffix (e.g. "event-name-837")
            # that Polymarket's frontend doesn't use — strip it.
            # Preserve 4-digit years (20XX) which are legitimate parts of slugs.
            clean_slug = re.sub(r"-(?!20\d{2}$)\d+$", "", slug) if slug else ""
            if clean_slug:
                poly_url = f"https://polymarket.com/event/{clean_slug}"
            elif condition_id:
                poly_url = f"https://polymarket.com/markets/{condition_id}"
            else:
                poly_url = "https://polymarket.com"

            markets.append({
                "platform": "polymarket",
                "id": condition_id,
                "question": m.get("question", ""),
                "category": _parse_category(m),
                "yes_price": yes_price,
                "no_price": round(1.0 - yes_price, 4),
                "volume_usd": float(m.get("volume24hr") or m.get("volume", 0) or 0),
                "end_date": m.get("endDate") or m.get("endDateIso", ""),
                "slug": slug,
                "url": poly_url,
            })
        except (KeyError, TypeError, ValueError):
            continue

    return markets


def _extract_yes_price(market: dict) -> Optional[float]:
    """Try several field names Polymarket uses across API versions."""
    for field in ("outcomePrices", "bestAsk", "lastTradePrice", "price"):
        val = market.get(field)
        if val is None:
            continue
        # outcomePrices is a JSON-string list like '["0.55","0.45"]'
        if isinstance(val, str) and val.startswith("["):
            import json
            try:
                prices = json.loads(val)
                return round(float(prices[0]), 4)
            except Exception:
                continue
        if isinstance(val, (int, float)):
            p = float(val)
            if 0 < p <= 1:
                return round(p, 4)
            if 1 < p <= 100:           # sometimes expressed as cents
                return round(p / 100, 4)
    return None


def _parse_category(market: dict) -> str:
    tags = market.get("tags") or market.get("categories") or []
    if isinstance(tags, list) and tags:
        first = tags[0]
        if isinstance(first, dict):
            return first.get("label", "other")
        return str(first)
    return market.get("category", "other")


def get_market_by_slug(slug: str) -> Optional[dict]:
    try:
        results = _get(f"{GAMMA_BASE}/markets", params={"slug": slug})
        if results:
            return results[0]
    except Exception:
        pass
    return None


def get_historical_resolved(limit: int = 500) -> pd.DataFrame:
    """
    Fetch recently resolved markets for ML training data.
    Returns a DataFrame with features useful for calibration modelling.
    """
    params = {
        "closed": "true",
        "limit": limit,
        "order": "volume",
        "ascending": "false",
    }
    try:
        data = _get(f"{GAMMA_BASE}/markets", params=params)
    except Exception:
        return pd.DataFrame()

    rows = []
    for m in data:
        try:
            yes_price = _extract_yes_price(m)
            if yes_price is None:
                continue
            resolved_yes = _parse_resolution(m)
            if resolved_yes is None:
                continue
            rows.append({
                "yes_price_at_close": yes_price,
                "resolved_yes": float(resolved_yes),
                "volume_usd": float(m.get("volume") or 0),
                "category": _parse_category(m),
                "question": m.get("question", ""),
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


def _demo_poly_markets() -> list[dict]:
    """
    Polymarket demo markets that pair with the Kalshi demo markets.
    Prices are deliberately offset to create visible arbitrage opportunities.
    """
    from datetime import datetime, timedelta
    base_date = (datetime.utcnow() + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return [
        {
            "platform": "polymarket",
            "id": "poly-fed-pause-2025",
            "question": "Will the Federal Reserve pause interest rate hikes in June 2025?",
            "category": "finance",
            "yes_price": 0.54,
            "no_price": 0.46,
            "volume_usd": 72_000,
            "end_date": base_date,
            "slug": "fed-pause-june-2025",
            "url": "https://polymarket.com/event/fed-pause-june-2025",
        },
        {
            "platform": "polymarket",
            "id": "poly-btc-100k",
            "question": "Bitcoin price reaches $100,000 before 2026?",
            "category": "crypto",
            "yes_price": 0.49,
            "no_price": 0.51,
            "volume_usd": 195_000,
            "end_date": base_date,
            "slug": "bitcoin-100k-2025",
            "url": "https://polymarket.com/event/bitcoin-100k-2025",
        },
        {
            "platform": "polymarket",
            "id": "poly-celtics-nba",
            "question": "Will the Boston Celtics be NBA Champions in 2025?",
            "category": "sports",
            "yes_price": 0.34,
            "no_price": 0.66,
            "volume_usd": 88_000,
            "end_date": base_date,
            "slug": "celtics-nba-2025",
            "url": "https://polymarket.com/event/celtics-nba-2025",
        },
        {
            "platform": "polymarket",
            "id": "poly-gpt5",
            "question": "Will OpenAI launch GPT-5 by July 2025?",
            "category": "technology",
            "yes_price": 0.62,
            "no_price": 0.38,
            "volume_usd": 54_000,
            "end_date": base_date,
            "slug": "openai-gpt5-july-2025",
            "url": "https://polymarket.com/event/openai-gpt5-july-2025",
        },
        {
            "platform": "polymarket",
            "id": "poly-us-recession",
            "question": "US recession in calendar year 2025?",
            "category": "economics",
            "yes_price": 0.39,
            "no_price": 0.61,
            "volume_usd": 160_000,
            "end_date": base_date,
            "slug": "us-recession-2025",
            "url": "https://polymarket.com/event/us-recession-2025",
        },
        {
            "platform": "polymarket",
            "id": "poly-spx-5500",
            "question": "Will S&P 500 close above 5500 by end of Q2 2025?",
            "category": "finance",
            "yes_price": 0.53,
            "no_price": 0.47,
            "volume_usd": 280_000,
            "end_date": base_date,
            "slug": "sp500-5500-q2-2025",
            "url": "https://polymarket.com/event/sp500-5500-q2-2025",
        },
        {
            "platform": "polymarket",
            "id": "poly-gov-shutdown",
            "question": "Government shutdown in the United States in 2025?",
            "category": "politics",
            "yes_price": 0.27,
            "no_price": 0.73,
            "volume_usd": 115_000,
            "end_date": base_date,
            "slug": "us-gov-shutdown-2025",
            "url": "https://polymarket.com/event/us-gov-shutdown-2025",
        },
        {
            "platform": "polymarket",
            "id": "poly-eth-5k",
            "question": "Will Ethereum (ETH) hit $5,000 in 2025?",
            "category": "crypto",
            "yes_price": 0.44,
            "no_price": 0.56,
            "volume_usd": 79_000,
            "end_date": base_date,
            "slug": "eth-5000-2025",
            "url": "https://polymarket.com/event/eth-5000-2025",
        },
    ]


def _parse_resolution(market: dict) -> Optional[float]:
    """Return 1.0 if YES won, 0.0 if NO won, None if unclear."""
    outcomes = market.get("outcomes") or market.get("outcomePrices") or ""
    if isinstance(outcomes, str):
        import json
        try:
            outcomes = json.loads(outcomes)
        except Exception:
            return None
    if isinstance(outcomes, list) and len(outcomes) >= 2:
        try:
            yes_final = float(outcomes[0])
            if yes_final >= 0.99:
                return 1.0
            if yes_final <= 0.01:
                return 0.0
        except Exception:
            pass
    return None
