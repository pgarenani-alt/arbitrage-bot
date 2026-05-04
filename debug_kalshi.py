"""
Kalshi API diagnostic — run this to see exactly what the API returns.
Usage:  python debug_kalshi.py <your-api-key>
"""
import sys, os, requests, json

key = sys.argv[1] if len(sys.argv) > 1 else os.getenv("KALSHI_API_KEY", "")
if not key:
    print("Usage: python debug_kalshi.py <api-key>")
    sys.exit(1)

ENDPOINTS = [
    ("trading.kalshi.com — kalshi-access-token", "https://trading.kalshi.com/trade-api/v2/markets",       {"kalshi-access-token": key}),
    ("trading.kalshi.com — Bearer",              "https://trading.kalshi.com/trade-api/v2/markets",       {"Authorization": f"Bearer {key}"}),
    ("elections.kalshi   — kalshi-access-token", "https://api.elections.kalshi.com/trade-api/v2/markets", {"kalshi-access-token": key}),
    ("elections.kalshi   — Bearer",              "https://api.elections.kalshi.com/trade-api/v2/markets", {"Authorization": f"Bearer {key}"}),
]

PARAMS = {"status": "open", "limit": 5}

for label, url, extra_headers in ENDPOINTS:
    headers = {"Content-Type": "application/json", "User-Agent": "ArbitrageBot/1.0", **extra_headers}
    try:
        r = requests.get(url, params=PARAMS, headers=headers, timeout=8)
        body = r.text[:400]
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  URL    : {r.url}")
        print(f"  Status : {r.status_code}")
        print(f"  Body   : {body}")
        if r.status_code == 200:
            data = r.json()
            markets = data.get("markets") or []
            print(f"  Markets returned: {len(markets)}")
            if markets:
                print(f"  First : {markets[0].get('title','')[:70]}")
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  ERROR: {e}")
