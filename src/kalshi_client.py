"""
Kalshi API client.
Uses Kalshi Trade API v2 — https://api.elections.kalshi.com/trade-api/v2

Authentication: RSA-PSS request signing (not a simple Bearer token).
Each request is signed with a private RSA key; the signature covers:
    timestamp_ms + HTTP_METHOD + /path

Required credentials (both needed for live mode):
    KALSHI_KEY_ID      — the access-key ID from the Kalshi dashboard
    KALSHI_PRIVATE_KEY — PEM-encoded RSA private key (can be multi-line)

Falls back to synthetic demo data derived from live Polymarket prices when
credentials are absent or the live API is unreachable.
"""
import os
import re
import base64
import time
import random
import requests
from typing import Optional

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
HEADERS_BASE = {
    "Content-Type": "application/json",
    "User-Agent":   "ArbitrageBot/1.0",
}

KALSHI_FEE_RATE = 0.07   # 7 % of winnings on winning side


# ─────────────────────────────────────────────────────────────────────────────
# RSA-PSS signing
# ─────────────────────────────────────────────────────────────────────────────

def _load_private_key(pem_text: str):
    """Parse a PEM private key from a string (supports \\n literals)."""
    from cryptography.hazmat.primitives import serialization
    # Accept both real newlines and escaped \n (common when stored in env vars)
    pem_text = pem_text.replace("\\n", "\n").strip()
    return serialization.load_pem_private_key(pem_text.encode(), password=None)


def _sign_headers(key_id: str, private_key, method: str, path: str) -> dict:
    """Return the three Kalshi auth headers for one request."""
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
    """
    Resolve (key_id, private_key) from arguments or environment variables.
    Returns (None, None) if credentials are incomplete.
    """
    kid = key_id         or os.getenv("KALSHI_KEY_ID",      "")
    pem = private_key_pem or os.getenv("KALSHI_PRIVATE_KEY", "")
    if not kid or not pem:
        return None, None
    try:
        return kid, _load_private_key(pem)
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helper
# ─────────────────────────────────────────────────────────────────────────────

def _get(path: str, params: dict = None,
         key_id: str = None, private_key=None, retries: int = 3) -> dict:
    url      = f"{KALSHI_BASE}{path}"
    last_err = None
    for attempt in range(retries):
        try:
            headers = _sign_headers(key_id, private_key, "GET", path)
            r = requests.get(url, params=params, headers=headers, timeout=10)
            if not r.ok:
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code in (401, 403):
                    raise RuntimeError(f"Kalshi auth failed — {last_err}")
                time.sleep(1.5 ** attempt)
                continue
            return r.json()
        except RuntimeError:
            raise
        except requests.RequestException as e:
            last_err = str(e)
            if attempt == retries - 1:
                raise RuntimeError(f"Kalshi request failed — {last_err}")
            time.sleep(1.5 ** attempt)
    raise RuntimeError(f"Kalshi request failed — {last_err}")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_active_markets(limit: int = 100,
                       key_id: str = None,
                       private_key_pem: str = None,
                       # legacy param kept for backward compat — ignored
                       api_key: str = None) -> list[dict]:
    """
    Fetch open Kalshi markets.
    Falls back to synthetic demo data when credentials are absent or
    the live API is unreachable (DNS error, auth failure, timeout, etc.).
    """
    kid, pk = _get_credentials(key_id, private_key_pem)
    if kid and pk:
        try:
            data    = _get("/markets", params={"status": "open", "limit": limit},
                           key_id=kid, private_key=pk)
            raw     = data.get("markets") or []
            markets = [m for m in (_normalise(r) for r in raw) if m]
            if markets:
                return markets
        except Exception:
            pass   # fall through to demo

    return _demo_markets_from_polymarket(limit)


# ─────────────────────────────────────────────────────────────────────────────
# Normalise a raw Kalshi market dict
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(m: dict) -> Optional[dict]:
    try:
        # v2 returns prices as dollar strings ("0.6500") or integer cents
        def _parse_price(val):
            if val is None:
                return 0.0
            f = float(val)
            return f if f <= 1.0 else f / 100   # cents → dollars

        yes_bid   = _parse_price(m.get("yes_bid"))
        yes_ask   = _parse_price(m.get("yes_ask"))
        yes_price = round((yes_bid + yes_ask) / 2, 4) if yes_ask else yes_bid

        if yes_price <= 0:
            return None

        vol = m.get("volume_24h") or m.get("volume") or 0

        return {
            "platform":   "kalshi",
            "id":         m.get("ticker", m.get("id", "")),
            "question":   m.get("title", ""),
            "category":   m.get("category", "other"),
            "yes_price":  yes_price,
            "no_price":   round(1.0 - yes_price, 4),
            "volume_usd": float(vol),
            "end_date":   m.get("close_time", ""),
            "slug":       m.get("ticker", ""),
            "url":        f"https://kalshi.com/markets/{m.get('ticker', '')}",
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Demo mode — synthesise Kalshi markets from live Polymarket data
# ─────────────────────────────────────────────────────────────────────────────

def _demo_markets_from_polymarket(limit: int = 100) -> list[dict]:
    """
    Pull live markets from Polymarket and synthesise a matching Kalshi market
    for each one with a small deterministic price offset.
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
                "question":   m.get("question", ""),
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
