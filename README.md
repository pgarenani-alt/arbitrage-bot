# Prediction Markets Arbitrage Bot
**Advanced Machine Learning — Final Project**

Detects price discrepancies between [Kalshi](https://kalshi.com) and [Polymarket](https://polymarket.com), scores them with an ML model, and generates AI-powered trade recommendations.

**Live demo:** https://arbitrage-bot-lwnpfz7g5fv23zungbeypy.streamlit.app

---

## Quick Start (Local)

```bash
# 1. Clone / open the project folder
cd arbitrage-bot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy env template and add your Anthropic key
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY (required for AI matching + recommendations)

# 4. Train the ML model  ← run this once
python scripts/train_model.py

# 5. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## API Keys

The live demo at the link above is fully configured — no API key needed to use it.

If running locally:

| Key | Required? | Where to get it |
|-----|-----------|-----------------|
| `ANTHROPIC_API_KEY` | **Yes** (for Claude AI matching + trade analysis) | [console.anthropic.com](https://console.anthropic.com) |

**No Kalshi or Polymarket API key is needed.** Both platforms expose public market data endpoints with no authentication required for read-only access.

---

## Project Structure

```
arbitrage-bot/
├── app.py                        # Streamlit UI (main entry point)
├── src/
│   ├── kalshi_client.py          # Kalshi public REST API + synthetic demo markets
│   ├── polymarket_client.py      # Polymarket Gamma API client
│   ├── market_matcher.py         # Three-tier AI matching pipeline (Component B)
│   └── arbitrage_engine.py       # Arbitrage detection + ML scoring (Component C)
├── scripts/
│   └── train_model.py            # ML training script (Component A)
├── models/                       # Saved model artifacts (committed — no retraining needed)
│   ├── arbitrage_scorer.pkl
│   ├── feature_names.pkl
│   ├── category_map.pkl
│   └── training_report.txt
├── requirements.txt
└── .env.example
```

---

## How It Works

### A. ML Model (Component A — 30 pts)
- **Data**: ~1,000 resolved markets fetched from Polymarket's public Gamma API
- **Features**: `yes_price`, `log_volume`, `days_to_resolution`, `category_enc`, `price_confidence`, `extreme_price`, `log_vol_x_price`
- **Target**: `profitable_arb` — 1 if the market's mid-price deviates enough from 0.5 to signal a misprice
- **Model**: XGBoost classifier with isotonic probability calibration
- **Evaluation**: ROC-AUC 0.765, 5-fold cross-validation (see `models/training_report.txt`)

### B. AI Component (Component B — 25 pts)

Three-tier matching pipeline — each Kalshi market is matched to its Polymarket counterpart using:

1. **Tier 1 — Claude Haiku** (`claude-haiku-4-5-20251001`): LLM semantic matching. Given a Kalshi question and up to 10 Polymarket candidates, Claude picks the best match and rates confidence (high/medium/low). Handles cross-platform rephrasing (e.g. "Will X happen?" vs "X by end of year?").

2. **Tier 2 — Sentence Embeddings + Cosine Similarity** (`all-MiniLM-L6-v2`): Each question is encoded into a 384-dimensional dense vector using a sentence-transformer model. Cosine similarity between the Kalshi embedding and each Polymarket candidate is computed as a dot product of L2-normalised vectors — so cosine similarity reduces to a simple dot product. A score ≥ 0.65 qualifies as a match (≥ 0.80 = high confidence). This tier directly applies the sentence embedding and cosine similarity concepts from the course.

3. **Tier 3 — Jaccard Keyword Overlap**: Classical NLP fallback. Stopwords are removed, then token overlap (intersection / union) ≥ 0.25 is used to find a match for any markets not handled by Tiers 1–2.

Claude Sonnet (`claude-sonnet-4-6`) also generates a plain-English trade recommendation for each opportunity, covering trade instructions, expected profit, and key risks.

### C. Decision Output (Component C — 25 pts)
- Ranked table of live arbitrage opportunities with expected profit %, ML confidence, days to resolution, and match quality
- Detailed per-opportunity view: exact trade instructions (buy YES on Kalshi, buy NO on Polymarket, etc.), combined cost, gross/net profit after fees
- Interactive ML scorer: enter any hypothetical price pair to get a model confidence score without re-running the full scan
- Match pipeline stats displayed on every scan: how many markets were matched by Claude vs embeddings (cosine similarity) vs keyword fallback

---

## Arbitrage Logic

A risk-free arbitrage exists when you can buy **both** YES on one platform and NO on the other for a combined cost < $1.00:

```
Example: Kalshi YES = 0.45 and Polymarket YES = 0.55
  → Buy YES on Kalshi     @ 0.45
  → Buy NO  on Polymarket @ 0.45   (= 1 - 0.55)
  Combined cost = 0.90
  Payout always = $1.00 (one side always wins)
  Gross profit  = $0.10 = 10%
```

After fees (~4–5% round-trip), net profit ≈ 5–6%. The engine caps gross profit at 20% — anything higher almost certainly means two markets were incorrectly matched.

---

## Deployment (Streamlit Cloud)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set `app.py` as the main file
4. Add `ANTHROPIC_API_KEY` as a Secret
5. Deploy — the model is pre-trained and committed to `models/`

---

## Data Sources

- **Polymarket**: `https://gamma-api.polymarket.com/markets` — public, no auth required
- **Kalshi**: [Trade API v2](https://trading-api.readme.io/reference/getmarkets) — public REST API, no auth required for market data. Note: REST APIs cannot be opened in a browser like a website — the app calls specific endpoints programmatically in the background. To verify the API is live, open this in a browser: [`/events?status=open&limit=5`](https://external-api.kalshi.com/trade-api/v2/events?status=open&limit=200) — it will return real Kalshi market data as JSON.
- **Training data**: ~1,000 resolved Polymarket markets fetched via the Gamma API
