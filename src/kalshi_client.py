"""
Kalshi API client.
Uses Kalshi Trade API v2 — https://external-api.kalshi.com/trade-api/v2

Public market data (GET /markets) requires NO authentication.
Trading endpoints require RSA-PSS request signing, but we don't trade here.

Falls back to synthetic demo data (derived from live Polymarket prices) only
if the public Kalshi endpoint is unreachable.
"""
import os
import re
import base64
import time
import random
import requests
from typing import Optional

# ── Endpoints ─────────────────────────────────────────────────────────────────
# Primary: public REST API, no auth required for market data
KALSHI_PUBLIC_BASE = "https://external-api.kalshi.com/trade-api/v2"

# Kept for completeness; used only when RSA credentials are provided
KALSHI_AUTH_BASE   = "https://api.elections.kalshi.com/trade-api/v2"

HEADERS_BASE = {
    "Content-Type": "application/json",
    "User-Agent":   "ArbitrageBot/1.0",
}

KALSHI_FEE_RATE = 0.07   # 7 % of winnings on winning side


# ─────────────────────────────────────────────────────────────────────────────
# Public (no-auth) HTTP helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_public(path: str, params: dict = None, retries: int = 3) -> dict:
    """GET against the public Kalshi endpoint — no authentication headers."""
    url      = f"{KALSHI_PUBLIC_BASE}{path}"
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS_BASE, timeout=10)
            if not r.ok:
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                time.sleep(1.5 ** attempt)
                continue
            return r.json()
        except requests.RequestException as e:
            last_err = str(e)
            if attempt < retries - 1:
                time.sleep(1.5 ** attempt)
    raise RuntimeError(f"Kalshi public request failed — {last_err}")


# ─────────────────────────────────────────────────────────────────────────────
# RSA-PSS signing (kept for optional authenticated endpoints)
# ─────────────────────────────────────────────────────────────────────────────

def _load_private_key(pem_text: str):
    """Parse a PEM private key from a string (supports \\n literals)."""
    from cryptography.hazmat.primitives import serialization
    pem_text = pem_text.replace("\\n", "\n").strip()
    return serialization.load_pem_private_key(pem_text.encode(), password=None)


def _sign_headers(key_id: str, private_key, method: str, path: str) -> dict:
    """Return the three Kalshi auth headers for one signed request."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    timestamp_ms = str(int(time.time() * 1000))
    message      = f"{timestamp_ms}{method.upper()}{path}".encode()
    signature    = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return {
        **HEADERS_BASE,
        "KALSHI-ACCESS-KEY":       key_id,
        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
    }


def _get_credentials(key_id: str = None, private_key_pem: str = None):
    kid = key_id          or os.getenv("KALSHI_KEY_ID",      "")
    pem = private_key_pem or os.getenv("KALSHI_PRIVATE_KEY", "")
    if not kid or not pem:
        return None, None
    try:
        return kid, _load_private_key(pem)
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_active_markets(limit: int = 100,
                       key_id: str = None,
                       private_key_pem: str = None,
                       api_key: str = None) -> list[dict]:
    """
    Fetch real Kalshi markets from the public REST API (no auth needed).
    Falls back to synthetic demo data ONLY if the public API is unreachable.
    Synthetic data is intentionally avoided when the live API works because
    it produces mismatched pairs with nonsensical price discrepancies.
    """
    try:
        live = _fetch_live_markets(limit)
        if live:
            return live
    except Exception:
        pass

    # True fallback — public API completely unreachable
    return _demo_markets_from_polymarket(limit)


def _fetch_live_markets(limit: int) -> list[dict]:
    """
    Fetch real binary Kalshi markets via the public events + markets endpoints.
    Uses concurrent requests so the full fetch stays under ~8 seconds.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Fetch a page of open events (category filter is ignored by the API,
    # so just grab a flat list and take whatever we get)
    try:
        data   = _get_public("/events", params={"status": "open", "limit": min(limit * 3, 200)})
        events = data.get("events") or []
    except Exception:
        return []

    if not events:
        return []

    def fetch_event_markets(event: dict) -> list[dict]:
        try:
            url = f"{KALSHI_PUBLIC_BASE}/markets"
            r   = requests.get(url,
                               params={"event_ticker": event["event_ticker"]},
                               headers=HEADERS_BASE, timeout=5)
            raw = r.json().get("markets", []) if r.ok else []
            for m in raw:
                m.setdefault("_category", event.get("category", "other"))
            return raw
        except Exception:
            return []

    all_raw: list[dict] = []
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(fetch_event_markets, ev) for ev in events]
        for future in as_completed(futures):
            all_raw.extend(future.result())

    markets = [m for m in (_normalise(r) for r in all_raw) if m]
    markets.sort(key=lambda m: m["volume_usd"], reverse=True)
    return markets[:limit]


# ─────────────────────────────────────────────────────────────────────────────
# Normalise a raw Kalshi market dict
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(m: dict) -> Optional[dict]:
    """
    Map a raw Kalshi API market dict to the app's standard market schema.

    The public API returns prices in the *_dollars fields as dollar strings
    ("0.6500"). Mid-price = (bid + ask) / 2; falls back to last_price_dollars
    if quotes are absent.
    """
    try:
        def _parse(val):
            if val is None:
                return 0.0
            f = float(val)
            # API says response_price_units = usd_cent but *_dollars fields
            # are already in [0,1] dollar format — no conversion needed.
            return f if f <= 1.0 else f / 100

        yes_bid  = _parse(m.get("yes_bid_dollars"))
        yes_ask  = _parse(m.get("yes_ask_dollars"))
        last     = _parse(m.get("last_price_dollars"))

        if yes_bid > 0 and yes_ask > 0:
            yes_price = round((yes_bid + yes_ask) / 2, 4)
        elif yes_ask > 0:
            yes_price = yes_ask
        elif yes_bid > 0:
            yes_price = yes_bid
        elif last > 0:
            yes_price = last
        else:
            return None   # no price data — skip

        if not (0.03 <= yes_price <= 0.97):
            return None   # near-certain or zero — not useful for arb

        # volume_fp is total volume; volume_24h_fp is 24-hour
        vol = float(m.get("volume_24h_fp") or m.get("volume_fp") or 0)

        # category comes from the event (injected as _category in _fetch_live_markets)
        category = (m.get("_category") or m.get("category") or "other").lower()

        ticker = m.get("ticker", "")
        return {
            "platform":   "kalshi",
            "id":         ticker,
            "question":   m.get("title", ""),
            "category":   category,
            "yes_price":  yes_price,
            "no_price":   round(1.0 - yes_price, 4),
            "volume_usd": vol,
            "end_date":   m.get("close_time") or m.get("expiration_time", ""),
            "slug":       ticker,
            "url":        f"https://kalshi.com/markets/{ticker}",
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Demo mode — synthesise Kalshi markets from live Polymarket data
# ─────────────────────────────────────────────────────────────────────────────

def _kalshi_rephrase(question: str) -> str:
    """
    Convert a Polymarket-style question into Kalshi-style phrasing.
    Kalshi questions drop the leading "Will" and use a more direct format,
    creating realistic cross-platform phrasing differences for the AI
    matching pipeline to resolve.
    """
    q = question.strip()
    if not q:
        return q
    q_work = re.sub(r"^[Ww]ill\s+", "", q).rstrip("?").strip()
    if q_work:
        q_work = q_work[0].upper() + q_work[1:]
    return (q_work + "?") if q_work else question


def _demo_markets_from_polymarket(limit: int = 100) -> list[dict]:
    """
    Pull live markets from Polymarket and synthesise a matching Kalshi market
    for each one with a small deterministic price offset.
    Used only when the public Kalshi endpoint is unreachable.
    """
    import hashlib, json as _json

    try:
        r = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"active": "true", "closed": "false",
                    "limit": limit, "order": "volume24hr", "ascending": "false"},
            headers={"User-Agent": "ArbitrageBot/1.0"},
            timeout=10,
        )
        r.raise_for_status()
        poly_markets = r.json()
    except Exception:
        return []

    result = []
    for m in poly_markets:
        try:
            yes_p = None
            for field in ("outcomePrices", "lastTradePrice", "bestAsk", "price"):
                val = m.get(field)
                if val is None:
                    continue
                if isinstance(val, str) and val.startswith("["):
                    prices = _json.loads(val)
                    yes_p  = float(prices[0])
                    break
                p     = float(val)
                yes_p = p if p <= 1.0 else p / 100
                break
            if yes_p is None or not (0.03 < yes_p < 0.97):
                continue

            seed   = int(hashlib.md5(str(m.get("id", "")).encode()).hexdigest()[:8], 16)
            rng    = random.Random(seed)
            sign   = 1 if seed % 2 == 0 else -1
            offset = sign * rng.uniform(0.03, 0.08)
            k_yes  = round(min(max(yes_p + offset, 0.03), 0.97), 3)

            tags = m.get("tags") or []
            if isinstance(tags, list) and tags:
                t   = tags[0]
                cat = t.get("label", "other") if isinstance(t, dict) else str(t)
            else:
                cat = m.get("category", "other")

            slug         = m.get("slug") or ""
            condition_id = m.get("conditionId") or m.get("id") or ""
            clean_slug   = re.sub(r"-(?!20\d{2}$)\d+$", "", slug) if slug else ""
            if clean_slug:
                poly_url = f"https://polymarket.com/event/{clean_slug}"
            elif condition_id:
                poly_url = f"https://polymarket.com/markets/{condition_id}"
            else:
                poly_url = "https://polymarket.com"

            result.append({
                "platform":   "kalshi",
                "id":         f"DEMO-{condition_id or slug}",
                "question":   _kalshi_rephrase(m.get("question", "")),
                "category":   cat.lower(),
                "yes_price":  k_yes,
                "no_price":   round(1.0 - k_yes, 3),
                "volume_usd": float(m.get("volume24hr") or m.get("volume") or 0),
                "end_date":   m.get("endDate") or m.get("endDateIso") or "",
                "slug":       slug,
                "url":        poly_url,
                "is_demo":    True,
            })
        except Exception:
            continue

    return result
