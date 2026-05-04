# Prediction Markets Arbitrage Bot
**Applied Machine Learning — Final Project**

Detects price discrepancies between [Kalshi](https://kalshi.com) and [Polymarket](https://polymarket.com), scores them with an ML model, and generates AI-powered trade recommendations.

---

## Quick Start (Local)

```bash
# 1. Clone / open the project folder
cd arbitrage-bot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy env template and fill in your keys
cp .env.example .env
# Edit .env: add ANTHROPIC_API_KEY (required for AI features)

# 4. Train the ML model  ← run this once
python scripts/train_model.py

# 5. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## API Keys

| Key | Required? | Where to get it |
|-----|-----------|-----------------|
| `ANTHROPIC_API_KEY` | **Yes** (for AI matching + analysis) | [console.anthropic.com](https://console.anthropic.com) |
| `KALSHI_API_KEY` | Optional | [kalshi.com](https://kalshi.com) → API Settings |

Polymarket requires **no API key** — market data is public.

Without the Kalshi API key the app uses a realistic demo dataset of 8 markets.

---

## Project Structure

```
arbitrage-bot/
├── app.py                        # Streamlit UI (main entry point)
├── src/
│   ├── kalshi_client.py          # Kalshi REST API wrapper + demo data
│   ├── polymarket_client.py      # Polymarket Gamma API client
│   ├── market_matcher.py         # Claude-powered cross-platform matching (Component B)
│   └── arbitrage_engine.py       # Arbitrage detection + ML scoring (Component C)
├── scripts/
│   └── train_model.py            # ML training script (Component A)
├── models/                       # Saved model artifacts (created by train script)
│   ├── arbitrage_scorer.pkl
│   ├── feature_names.pkl
│   └── training_report.txt
├── requirements.txt
└── .env.example
```

---

## How It Works

### A. ML Model (Component A — 30 pts)
- **Data**: Historical resolved markets from Polymarket's public Gamma API
- **Features**: `yes_price`, `spread`, `combined_cost`, `price_level`, `log_volume`, `days_to_resolution`, `category_enc`, `is_near_boundary`, `days_log`
- **Target**: `profitable_arb` — 1 if the arbitrage would have been profitable after fees (combined cost < 0.96)
- **Model**: XGBoost classifier with isotonic probability calibration
- **Evaluation**: ROC-AUC, precision/recall, 5-fold cross-validation (reported in `models/training_report.txt`)

### B. AI Component (Component B — 25 pts)
Claude is used for two tasks:
1. **Market matching** (`claude-haiku-4-5-20251001`): Semantically pairs Kalshi and Polymarket markets that describe the same event (different wording, date formats, etc.)
2. **Trade recommendation** (`claude-sonnet-4-6`): Generates a plain-English analysis of each opportunity including trade instructions, risk factors, and event context

### C. Decision Output (Component C — 25 pts)
- Ranked table of live arbitrage opportunities with expected profit %, ML confidence, and match quality
- Detailed view per opportunity: exact trade instructions (buy X on Kalshi, buy Y on Polymarket), combined cost, gross/net profit
- Interactive ML scorer so the professor can input any hypothetical numbers

---

## Arbitrage Logic

A risk-free arbitrage exists when you can buy **both** YES on one platform and NO on the other for a combined cost < $1.00:

```
If Kalshi YES = 0.45 and Polymarket YES = 0.55:
  → Buy YES on Kalshi @ 0.45
  → Buy NO  on Polymarket @ 0.45   (= 1 - 0.55)
  Combined cost = 0.90
  Payout always = $1.00 (one side always wins)
  Gross profit  = $0.10 = 10%
```

After fees (~4% round-trip), net profit ≈ 6%.

---

## Deployment (Streamlit Cloud)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set `app.py` as the main file
4. Add `ANTHROPIC_API_KEY` (and optionally `KALSHI_API_KEY`) as Secrets
5. Deploy — the model is pre-trained and committed to `models/`

---

## Data Sources

- **Polymarket**: `https://gamma-api.polymarket.com/markets` — public, no auth
- **Kalshi**: `https://api.elections.kalshi.com/trade-api/v2/markets` — requires API key (free)
- **Training data**: Resolved Polymarket markets (500–1500 rows depending on API availability)
