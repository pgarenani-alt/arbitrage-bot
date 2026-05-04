"""Quick smoke test — run from project root."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.kalshi_client import get_active_markets as km
from src.polymarket_client import get_active_markets as pm
from src.arbitrage_engine import find_opportunities, opportunities_to_dataframe

print("=== Kalshi markets ===")
kalshi = km(limit=75)
print(f"  {len(kalshi)} markets returned")
if kalshi:
    print(f"  First: {kalshi[0]['question'][:70]}")

print("\n=== Polymarket markets ===")
poly = pm(limit=75)
print(f"  {len(poly)} markets returned")
if poly:
    print(f"  First: {poly[0]['question'][:70]}")

print("\n=== Arbitrage detection ===")
opps = find_opportunities(kalshi, poly, use_ai_matching=False, min_profit_pct=0.5)
print(f"  {len(opps)} opportunities found")
if opps:
    o = opps[0]
    print(f"  Best: {o['expected_profit_pct']:.2f}% gross | ML conf: {o['ml_confidence']:.0%}")
    print(f"  Trade: {o['kalshi_platform_action']} Kalshi | {o['poly_platform_action']} Polymarket")

print("\nAll imports and logic OK.")
