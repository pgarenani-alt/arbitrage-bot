"""
Prediction Markets Arbitrage Bot
Applied Machine Learning — Final Project
"""
import os
import sys
import time
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from src.kalshi_client      import get_active_markets as kalshi_markets
from src.polymarket_client  import get_active_markets as poly_markets
from src.arbitrage_engine   import find_opportunities, opportunities_to_dataframe

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "Prediction Markets Arbitrage Bot",
    page_icon   = "📊",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Resolve API keys before any UI elements reference them
# ─────────────────────────────────────────────────────────────────────────────

anthropic_key = (
    st.secrets.get("ANTHROPIC_API_KEY", "")
    if hasattr(st, "secrets") else ""
) or os.getenv("ANTHROPIC_API_KEY", "")
if anthropic_key:
    os.environ["ANTHROPIC_API_KEY"] = anthropic_key

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")

    if not anthropic_key:
        _typed_key = st.text_input(
            "Anthropic API Key",
            value="",
            type="password",
            help="Required for AI market matching and trade recommendations.",
        )
        if _typed_key:
            anthropic_key = _typed_key
            os.environ["ANTHROPIC_API_KEY"] = anthropic_key

    kalshi_key = st.text_input(
        "Kalshi API Key (optional)",
        value=os.getenv("KALSHI_API_KEY", ""),
        type="password",
        help="Leave blank to use demo market data.",
    )
    if kalshi_key:
        os.environ["KALSHI_API_KEY"] = kalshi_key

    st.divider()
    st.subheader("Filters")
    min_profit = st.slider("Min Gross Profit %", 0.5, 10.0, 1.5, 0.5)
    min_vol    = st.slider("Min Volume ($)", 0, 500_000, 10_000, 10_000,
                           format="$%d")
    use_ai     = st.checkbox("AI market matching (Claude)", value=bool(anthropic_key))
    n_markets  = st.slider("Markets to fetch per platform", 20, 150, 75, 5)

    st.divider()
    if kalshi_key:
        if st.button("🔬 Debug Kalshi API", use_container_width=True):
            with st.spinner("Testing Kalshi endpoints…"):
                report = _debug_kalshi(kalshi_key)
            st.session_state["kalshi_debug"] = report
        if "kalshi_debug" in st.session_state:
            with st.expander("Kalshi debug output", expanded=True):
                st.markdown(st.session_state["kalshi_debug"])

    st.divider()
    st.caption("Data sources: Kalshi Trade API v2 · Polymarket Gamma API")
    st.caption("AI: Anthropic Claude (market matching + analysis)")
    st.caption("ML: XGBoost calibrated classifier")

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("📊 Prediction Markets Arbitrage Bot")
st.caption(
    "Detects cross-platform price discrepancies between Kalshi and Polymarket, "
    "scores them with an ML model, and generates AI-powered trade recommendations."
)

_tab_scan, _tab_explain, _tab_model = st.tabs(
    ["🔍 Live Scan", "🤖 AI Analysis", "🧠 ML Model"]
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=120, show_spinner=False)
def _fetch_kalshi(n: int, _kalshi_key: str):
    # Separate cached function so a Kalshi error doesn't also bust the Poly cache
    return kalshi_markets(limit=n, api_key=_kalshi_key or None)

@st.cache_data(ttl=120, show_spinner=False)
def _fetch_poly(n: int):
    return poly_markets(limit=n)


@st.cache_data(ttl=120, show_spinner=False)
def _find_opps(kalshi_json: str, poly_json: str, use_ai: bool, min_profit: float):
    import json
    km = json.loads(kalshi_json)
    pm = json.loads(poly_json)
    return find_opportunities(km, pm, use_ai_matching=use_ai, min_profit_pct=min_profit)


def _confidence_color(conf: float) -> str:
    if conf >= 0.70:
        return "🟢"
    if conf >= 0.45:
        return "🟡"
    return "🔴"


def _profit_gauge(profit_pct: float, ml_conf: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=profit_pct,
        number={"suffix": "%", "font": {"size": 36}},
        title={"text": "Gross Profit Potential", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 10], "tickwidth": 1},
            "bar":  {"color": "#2196F3"},
            "steps": [
                {"range": [0, 2],  "color": "#ffcccc"},
                {"range": [2, 5],  "color": "#ffffcc"},
                {"range": [5, 10], "color": "#ccffcc"},
            ],
            "threshold": {
                "line": {"color": "green", "width": 3},
                "thickness": 0.8,
                "value": profit_pct * ml_conf,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def _price_bar(km: dict, pm: dict) -> go.Figure:
    categories = ["Kalshi YES", "Kalshi NO", "Polymarket YES", "Polymarket NO"]
    prices     = [km["kalshi_yes_price"], 1 - km["kalshi_yes_price"],
                  pm["poly_yes_price"],   1 - pm["poly_yes_price"]]
    colors     = ["#1565C0", "#90CAF9", "#E65100", "#FFCC80"]

    fig = go.Figure(go.Bar(
        x=categories, y=prices, marker_color=colors,
        text=[f"{p:.3f}" for p in prices], textposition="outside",
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color="grey", annotation_text="Fair value 0.50")
    fig.update_layout(
        title="Price Comparison",
        yaxis=dict(range=[0, 1.1], title="Implied Probability"),
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _debug_kalshi(api_key: str) -> str:
    """Hit several Kalshi URL/header combos and return a readable report."""
    import requests as _req
    lines = []
    combos = [
        ("trading.kalshi.com + kalshi-access-token",
         "https://trading.kalshi.com/trade-api/v2/markets",
         {"kalshi-access-token": api_key}),
        ("trading.kalshi.com + Bearer",
         "https://trading.kalshi.com/trade-api/v2/markets",
         {"Authorization": f"Bearer {api_key}"}),
        ("elections.kalshi   + kalshi-access-token",
         "https://api.elections.kalshi.com/trade-api/v2/markets",
         {"kalshi-access-token": api_key}),
        ("elections.kalshi   + Bearer",
         "https://api.elections.kalshi.com/trade-api/v2/markets",
         {"Authorization": f"Bearer {api_key}"}),
    ]
    base_headers = {"Content-Type": "application/json", "User-Agent": "ArbitrageBot/1.0"}
    for label, url, auth in combos:
        try:
            r = _req.get(url, params={"status": "open", "limit": 3},
                         headers={**base_headers, **auth}, timeout=8)
            body = r.text[:500]
            n = len((r.json() if r.ok else {}).get("markets") or [])
            lines.append(f"**{label}**\n`{r.status_code}` — {n} markets\n```\n{body}\n```")
        except Exception as e:
            lines.append(f"**{label}**\nERROR: {e}")
    return "\n\n---\n\n".join(lines)


def _combined_cost_chart(opps: list[dict]) -> go.Figure:
    if not opps:
        return go.Figure()
    df = pd.DataFrame({
        "Event":        [o["kalshi_question"][:40] + "…" for o in opps],
        "Combined Cost": [o["combined_cost"] for o in opps],
        "ML Confidence": [o["ml_confidence"] for o in opps],
        "Gross Profit %": [o["expected_profit_pct"] for o in opps],
    })
    fig = px.scatter(
        df, x="Combined Cost", y="Gross Profit %",
        size="ML Confidence", color="ML Confidence",
        hover_name="Event",
        color_continuous_scale="RdYlGn",
        title="Opportunities: Cost vs Profit (bubble size = ML confidence)",
        labels={"Combined Cost": "Combined Entry Cost ($1 payout)"},
    )
    fig.add_vline(x=0.97, line_dash="dash", line_color="red",
                  annotation_text="Break-even (4% fees)")
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Live Scan
# ─────────────────────────────────────────────────────────────────────────────

with _tab_scan:
    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        run_scan = st.button("🔄 Scan Markets", type="primary", use_container_width=True)
    with col_status:
        status_placeholder = st.empty()

    if run_scan or "opportunities" not in st.session_state:
        with st.spinner("Fetching markets from Kalshi & Polymarket…"):
            # Fetch separately so a Kalshi error is visible and doesn't block Poly
            km_list = []
            try:
                km_list = _fetch_kalshi(n_markets, kalshi_key)
            except Exception as e:
                st.error(f"Kalshi fetch error: {e}")

            pm_list = []
            try:
                pm_list = _fetch_poly(n_markets)
            except Exception as e:
                st.error(f"Polymarket fetch error: {e}")

            source = "live API" if kalshi_key else "demo (no key)"
            status_placeholder.success(
                f"Fetched {len(km_list)} Kalshi ({source}) + {len(pm_list)} Polymarket markets"
            )

        if km_list or pm_list:
            with st.spinner("Matching markets and detecting arbitrage…"):
                import json
                opps = _find_opps(
                    json.dumps(km_list), json.dumps(pm_list),
                    use_ai and bool(anthropic_key),
                    min_profit,
                )
            # filter by min volume
            opps = [o for o in opps if o.get("min_volume_usd", 0) >= min_vol]
            st.session_state["opportunities"] = opps
            st.session_state["km_list"]       = km_list
            st.session_state["pm_list"]       = pm_list

    opps = st.session_state.get("opportunities", [])

    # ── Summary metrics ──────────────────────────────────────────────────────
    if opps:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Opportunities Found", len(opps))
        m2.metric("Best Gross Profit",   f"{max(o['expected_profit_pct'] for o in opps):.2f}%")
        m3.metric("Avg ML Confidence",   f"{np.mean([o['ml_confidence'] for o in opps]):.0%}")
        m4.metric("High Confidence (≥70%)", sum(o['ml_confidence'] >= 0.70 for o in opps))

        st.plotly_chart(_combined_cost_chart(opps), use_container_width=True)

    # ── Opportunities table ───────────────────────────────────────────────────
    df_display = opportunities_to_dataframe(opps)

    if df_display.empty:
        st.info(
            "No arbitrage opportunities found above the current filters. "
            "Try lowering the Min Gross Profit % or Min Volume, or rescan."
        )
    else:
        st.subheader(f"Detected Opportunities ({len(opps)})")
        display_cols = [c for c in df_display.columns if not c.startswith("_")]

        # colour-code ML confidence
        def style_conf(v):
            try:
                f = float(v.strip("%")) / 100
                if f >= 0.70:
                    return "background-color:#c8f7c5;color:#1a1a1a"
                if f >= 0.45:
                    return "background-color:#fff9c4;color:#1a1a1a"
                return "background-color:#ffcdd2;color:#1a1a1a"
            except Exception:
                return "color:#1a1a1a"

        styled = df_display[display_cols].style.map(style_conf, subset=["ML Confidence"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Row selector
        selected_idx = st.selectbox(
            "Select an opportunity to inspect →",
            options=range(len(opps)),
            format_func=lambda i: opps[i]["kalshi_question"][:80],
        )
        st.session_state["selected_opp"] = opps[selected_idx]

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — AI Analysis
# ─────────────────────────────────────────────────────────────────────────────

with _tab_explain:
    opp = st.session_state.get("selected_opp")

    if not opp:
        st.info("Run a scan on the Live Scan tab and select an opportunity.")
    else:
        st.subheader(opp["kalshi_question"])

        # ── Trade card ───────────────────────────────────────────────────────
        col_l, col_r = st.columns(2)

        with col_l:
            st.plotly_chart(_price_bar(opp, opp), use_container_width=True)

        with col_r:
            st.plotly_chart(
                _profit_gauge(opp["expected_profit_pct"], opp["ml_confidence"]),
                use_container_width=True
            )

        # ── Trade instructions ───────────────────────────────────────────────
        st.markdown("### Trade Instructions")
        trade_cols = st.columns(2)

        _kalshi_is_demo = opp.get("kalshi_url", "").startswith("https://polymarket.com")
        _kalshi_link_label = "View on Polymarket (demo mode) ↗" if _kalshi_is_demo else "Open on Kalshi ↗"

        with trade_cols[0]:
            st.markdown(
                f"""
                <div style="background:#e3f2fd;padding:16px;border-radius:8px">
                <h4 style="margin:0;color:#1565c0">🏦 Kalshi</h4>
                <p style="font-size:20px;margin:8px 0;color:#111111">
                  <b>{opp['kalshi_platform_action']}</b> @ <b>{opp['kalshi_trade_price']:.3f}</b>
                </p>
                <a href="{opp['kalshi_url']}" target="_blank" style="color:#1565c0">{_kalshi_link_label}</a>
                </div>
                """, unsafe_allow_html=True
            )

        with trade_cols[1]:
            st.markdown(
                f"""
                <div style="background:#fff3e0;padding:16px;border-radius:8px">
                <h4 style="margin:0;color:#e65100">🔮 Polymarket</h4>
                <p style="font-size:20px;margin:8px 0;color:#111111">
                  <b>{opp['poly_platform_action']}</b> @ <b>{opp['poly_trade_price']:.3f}</b>
                </p>
                <a href="{opp['poly_url']}" target="_blank" style="color:#e65100">Open on Polymarket ↗</a>
                </div>
                """, unsafe_allow_html=True
            )

        st.markdown("---")

        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("Combined Cost",    f"${opp['combined_cost']:.3f}")
        ec2.metric("Gross Profit",     f"{opp['expected_profit_pct']:.2f}%")
        ec3.metric("Net (after fees)", f"~{opp['net_profit_pct']:.2f}%")
        ec4.metric("ML Confidence",    f"{opp['ml_confidence']:.0%}",
                   delta=_confidence_color(opp["ml_confidence"]))

        st.markdown("---")

        # ── AI recommendation ─────────────────────────────────────────────
        st.markdown("### 🤖 AI Trade Analysis")

        if not anthropic_key:
            st.warning("Add an Anthropic API key in the sidebar to enable AI analysis.")
        else:
            if st.button("Generate AI Recommendation", type="primary"):
                with st.spinner("Asking Claude to analyse this opportunity…"):
                    try:
                        from src.market_matcher import generate_recommendation
                        rec = generate_recommendation(opp)
                        st.session_state["ai_recommendation"] = rec
                    except Exception as e:
                        st.session_state["ai_recommendation"] = f"Error: {e}"

            if "ai_recommendation" in st.session_state:
                st.markdown(st.session_state["ai_recommendation"])

        # ── Match metadata ──────────────────────────────────────────────────
        with st.expander("Match metadata"):
            st.write(f"**Kalshi question:** {opp['kalshi_question']}")
            st.write(f"**Polymarket question:** {opp['poly_question']}")
            st.write(f"**Match confidence:** {opp['match_confidence']}")
            st.write(f"**Reasoning:** {opp['match_reasoning']}")
            st.write(f"**Days to resolution:** {opp['days_to_resolution']}")
            st.write(f"**Min volume (USD):** ${opp['min_volume_usd']:,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — ML Model
# ─────────────────────────────────────────────────────────────────────────────

with _tab_model:
    st.subheader("🧠 ML Model: Arbitrage Opportunity Scorer")
    st.markdown(
        """
        **Component A** of the project — an XGBoost binary classifier trained on
        historical resolved Polymarket markets.

        **Task:** Given features of a detected price discrepancy, predict whether
        the opportunity will be profitable after trading fees.

        **Target variable:** `profitable_arb = 1` when `combined_cost < 0.96`
        (i.e., gross profit > 4%, enough to cover fees on both platforms).
        """
    )

    # Show training report if available
    report_path = os.path.join(os.path.dirname(__file__), "models", "training_report.txt")
    if os.path.exists(report_path):
        with open(report_path, encoding="latin-1") as f:
            report = f.read()
        with st.expander("Training Report", expanded=True):
            st.code(report)
    else:
        st.warning(
            "Model not trained yet.  Run `python scripts/train_model.py` to train it. "
            "The app will use a heuristic scorer in the meantime."
        )

    st.markdown("---")
    st.markdown("### Feature Descriptions")
    features_df = pd.DataFrame([
        ("yes_price",          "Kalshi YES price (0–1)",                                  "Direct price input"),
        ("spread",             "Absolute price difference between platforms",              "Larger = bigger apparent opportunity"),
        ("combined_cost",      "Sum of both trade prices",                                "< 1.0 = profitable before fees"),
        ("price_level",        "min(price, 1-price): how extreme the probability is",     "Near 0 = near-certain; near 0.5 = uncertain"),
        ("log_volume",         "log(1 + 24h trading volume USD)",                         "Liquidity proxy; low vol → hard to fill"),
        ("days_to_resolution", "Calendar days until market closes",                       "Longer window = more time for prices to converge"),
        ("category_enc",       "Encoded event category (politics, finance, crypto, …)",   "Some categories are better calibrated"),
        ("is_near_boundary",   "1 if YES price is between 0.45 and 0.55",                "High uncertainty → higher misquote risk"),
        ("days_log",           "log(1 + days_to_resolution)",                             "Smoothed time feature"),
    ], columns=["Feature", "Description", "Intuition"])
    st.dataframe(features_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Interactive Scorer")
    st.caption("Adjust sliders to see how the ML model would score a hypothetical opportunity.")

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        sim_yes      = st.slider("Kalshi YES price",        0.05, 0.95, 0.40, 0.01)
        sim_other    = st.slider("Polymarket YES price",    0.05, 0.95, 0.55, 0.01)
    with sc2:
        sim_vol      = st.slider("Market volume ($K)",      1, 500, 50, 1) * 1_000
        sim_days     = st.slider("Days to resolution",      1, 180, 30, 1)
    with sc3:
        sim_cat_name = st.selectbox("Category",
            ["Politics", "Finance", "Crypto", "Sports", "Technology", "Other"])
        cat_map_ui   = {"Politics":0,"Finance":1,"Crypto":2,"Sports":3,"Technology":4,"Other":7}
        sim_cat      = cat_map_ui[sim_cat_name]

    sim_combined = sim_yes + (1.0 - sim_other)
    sim_spread   = abs(sim_yes - sim_other)
    sim_profit   = max(0.0, 1.0 - sim_combined)

    import joblib as _jl
    model_path = os.path.join(os.path.dirname(__file__), "models", "arbitrage_scorer.pkl")
    if os.path.exists(model_path):
        model    = _jl.load(model_path)
        _log_vol = math.log1p(sim_vol)
        _price_conf  = _log_vol * (1 - 2 * abs(sim_yes - 0.5))
        _extreme     = int(sim_yes < 0.10 or sim_yes > 0.90)
        _log_vol_x_p = _log_vol * sim_yes
        x = np.array([[
            sim_yes,
            _log_vol,
            float(sim_days),
            float(sim_cat),
            _price_conf,
            float(_extreme),
            _log_vol_x_p,
        ]])
        try:
            sim_conf = float(model.predict_proba(x)[0][1])
        except Exception:
            sim_conf = sim_profit * 5
    else:
        sim_conf = min(sim_profit * 6, 0.95)

    res1, res2, res3 = st.columns(3)
    res1.metric("Combined Cost",   f"{sim_combined:.3f}")
    res2.metric("Gross Profit %",  f"{sim_profit * 100:.2f}%")
    res3.metric("ML Confidence",   f"{sim_conf:.0%}",
                delta=("✅ Trade" if sim_conf >= 0.6 else "⚠️ Risky"))

    st.plotly_chart(
        go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=sim_conf * 100,
            number={"suffix": "%"},
            delta={"reference": 60},
            title={"text": "Model Confidence: Profitable Arbitrage"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#1976D2"},
                "steps": [
                    {"range": [0, 40],  "color": "#ffcdd2"},
                    {"range": [40, 65], "color": "#fff9c4"},
                    {"range": [65, 100],"color": "#c8f7c5"},
                ],
                "threshold": {"line": {"color": "green","width": 4},
                              "thickness": 0.85, "value": 65},
            },
        )).update_layout(height=300, margin=dict(l=30,r=30,t=50,b=20)),
        use_container_width=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Built for Applied Machine Learning — Final Project · "
    "Kalshi + Polymarket Arbitrage Bot · "
    "ML: XGBoost | AI: Claude (Anthropic) | UI: Streamlit"
)
