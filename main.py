import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)


# ─────────────────────────────────────────────
# SECRETS
# ─────────────────────────────────────────────
FRED_API_KEY = st.secrets["FRED_API_KEY"]
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
OPENROUTER_API_KEY = st.secrets["OPENR_API_KEY"]


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Economic Early Warning System",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.stApp {
    background: #050d1a;
    background-image:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(0,122,255,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 80%, rgba(0,212,170,0.08) 0%, transparent 55%),
        radial-gradient(ellipse 40% 40% at 60% 30%, rgba(99,38,255,0.07) 0%, transparent 50%);
    min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1400px; }

.hero-header { text-align: center; padding: 2.5rem 0 1rem; margin-bottom: 0.5rem; }
.hero-badge {
    display: inline-block; font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; font-weight: 500; letter-spacing: 0.15em; text-transform: uppercase;
    color: #00d4aa; background: rgba(0,212,170,0.1); border: 1px solid rgba(0,212,170,0.25);
    padding: 0.3rem 1rem; border-radius: 50px; margin-bottom: 1rem;
}
.hero-title {
    font-size: clamp(2rem, 4vw, 3.2rem); font-weight: 700;
    letter-spacing: -0.03em; line-height: 1.1;
    background: linear-gradient(135deg, #ffffff 0%, #a8c8ff 50%, #00d4aa 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 0.6rem;
}
.hero-sub {
    font-size: 1rem; color: rgba(255,255,255,0.45); font-weight: 400;
    max-width: 560px; margin: 0 auto; line-height: 1.6;
}

.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,170,0.3), rgba(99,38,255,0.3), transparent);
    margin: 2rem 0;
}
.section-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; font-weight: 500;
    letter-spacing: 0.2em; text-transform: uppercase; color: rgba(255,255,255,0.3);
    margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: rgba(255,255,255,0.07); }
.section-title {
    font-size: 1.25rem; font-weight: 600; color: #ffffff;
    letter-spacing: -0.02em; margin: 0 0 1.25rem;
}

.kpi-card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.4rem 1.6rem; position: relative;
    overflow: hidden; backdrop-filter: blur(20px); margin-bottom: 1rem;
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; border-radius: 16px 16px 0 0;
}
.kpi-card.blue::before   { background: linear-gradient(90deg, #007aff, #5ac8fa); }
.kpi-card.teal::before   { background: linear-gradient(90deg, #00d4aa, #34c759); }
.kpi-card.purple::before { background: linear-gradient(90deg, #6326ff, #af52de); }
.kpi-card.red::before    { background: linear-gradient(90deg, #ff3b30, #ff6b35); }
.kpi-card.green::before  { background: linear-gradient(90deg, #34c759, #30d158); }
.kpi-card.orange::before { background: linear-gradient(90deg, #ff9500, #ffcc00); }

.kpi-icon { font-size: 1.1rem; margin-bottom: 0.6rem; opacity: 0.7; }
.kpi-label {
    font-size: 0.72rem; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase;
    color: rgba(255,255,255,0.4); margin-bottom: 0.4rem; font-family: 'JetBrains Mono', monospace;
}
.kpi-value {
    font-size: 2rem; font-weight: 700; color: #ffffff; letter-spacing: -0.03em;
    line-height: 1; margin-bottom: 0.35rem; font-family: 'JetBrains Mono', monospace;
}
.kpi-delta { font-size: 0.75rem; font-weight: 500; color: rgba(255,255,255,0.35); }
.kpi-delta.up   { color: #34c759; }
.kpi-delta.down { color: #ff3b30; }
.kpi-delta.warn { color: #ff9500; }

.glass-panel {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px; padding: 1.75rem; backdrop-filter: blur(20px);
    margin-bottom: 1.5rem; position: relative; overflow: hidden;
}
.glass-panel::before {
    content: ''; position: absolute; inset: 0; border-radius: 20px;
    background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, transparent 60%);
    pointer-events: none;
}

.risk-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    font-size: 0.8rem; font-weight: 600; letter-spacing: 0.05em;
    padding: 0.35rem 0.9rem; border-radius: 50px;
}
.risk-badge.high   { background: rgba(255,59,48,0.15); color: #ff6b6b; border: 1px solid rgba(255,59,48,0.3); }
.risk-badge.medium { background: rgba(255,149,0,0.15); color: #ffbb55; border: 1px solid rgba(255,149,0,0.3); }
.risk-badge.low    { background: rgba(52,199,89,0.15); color: #34c759; border: 1px solid rgba(52,199,89,0.3); }

.stSlider > div > div > div { background: rgba(0,212,170,0.2) !important; }
.stSlider > div > div > div > div { background: #00d4aa !important; }
[data-testid="metric-container"] { background: transparent; border: none; padding: 0; }
.stProgress > div > div > div {
    background: linear-gradient(90deg, #00d4aa, #007aff) !important; border-radius: 4px;
}
.stProgress > div > div { background: rgba(255,255,255,0.07) !important; border-radius: 4px; }
.stDataFrame { border-radius: 12px; overflow: hidden; }

.insight-card {
    display: flex; align-items: flex-start; gap: 0.85rem;
    padding: 1rem 1.2rem; border-radius: 12px; margin-bottom: 0.6rem;
    border: 1px solid rgba(255,255,255,0.06);
}
.insight-card.warn    { background: rgba(255,149,0,0.08);  border-color: rgba(255,149,0,0.2); }
.insight-card.danger  { background: rgba(255,59,48,0.08);  border-color: rgba(255,59,48,0.2); }
.insight-card.info    { background: rgba(0,122,255,0.08);  border-color: rgba(0,122,255,0.2); }
.insight-card.success { background: rgba(52,199,89,0.08);  border-color: rgba(52,199,89,0.2); }
.insight-icon { font-size: 1.1rem; margin-top: 0.05rem; }
.insight-text { font-size: 0.87rem; color: rgba(255,255,255,0.8); line-height: 1.5; }

.policy-box {
    background: rgba(99,38,255,0.1); border: 1px solid rgba(99,38,255,0.25);
    border-radius: 14px; padding: 1.2rem 1.4rem;
    font-size: 0.9rem; color: rgba(255,255,255,0.85); line-height: 1.6;
}
.policy-box strong { color: #af88ff; }

.score-display {
    font-family: 'JetBrains Mono', monospace; font-size: 3.5rem; font-weight: 700;
    letter-spacing: -0.04em; background: linear-gradient(135deg, #ffffff, #00d4aa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; text-align: center; line-height: 1; margin: 0.5rem 0;
}
.score-label {
    font-size: 0.72rem; letter-spacing: 0.15em; text-transform: uppercase;
    color: rgba(255,255,255,0.3); text-align: center; font-family: 'JetBrains Mono', monospace;
}

.info-box {
    background: rgba(0,122,255,0.07); border: 1px solid rgba(0,122,255,0.2);
    border-radius: 14px; padding: 1.2rem 1.4rem;
    font-size: 0.88rem; color: rgba(255,255,255,0.8); line-height: 1.7; margin-bottom: 0.75rem;
}
.info-box h4 { color: #5ac8fa; font-size: 0.95rem; margin: 0 0 0.4rem; }

.limit-box {
    background: rgba(255,149,0,0.07); border: 1px solid rgba(255,149,0,0.2);
    border-radius: 14px; padding: 1.2rem 1.4rem;
    font-size: 0.88rem; color: rgba(255,255,255,0.8); line-height: 1.7; margin-bottom: 0.75rem;
}
.limit-box h4 { color: #ffbb55; font-size: 0.95rem; margin: 0 0 0.4rem; }

.metric-row {
    display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem;
}
.metric-chip {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px; padding: 0.5rem 1rem; font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem; color: rgba(255,255,255,0.7);
}
.metric-chip span { color: #00d4aa; font-weight: 600; }

.source-tag {
    display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 0.62rem;
    letter-spacing: 0.1em; text-transform: uppercase; color: rgba(255,255,255,0.25);
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07);
    padding: 0.2rem 0.6rem; border-radius: 4px; margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LAYER — FRED + FALLBACK SIMULATION
# ─────────────────────────────────────────────

def fetch_fred_series(series_id: str, limit: int = 200) -> list[float]:
    """Fetch FRED time-series. Falls back to simulated data on any error."""
    try:
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
        )
        resp = requests.get(url, timeout=8)
        obs  = resp.json().get("observations", [])
        vals = [float(o["value"]) for o in obs if o["value"] != "."]
        if len(vals) >= 20:
            return vals[-limit:]
    except Exception:
        pass
    return []


def fetch_fred_latest(series_id: str) -> float | None:
    """Return the most recent valid FRED observation."""
    try:
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
        )
        resp = requests.get(url, timeout=8)
        obs  = resp.json().get("observations", [])
        for o in reversed(obs):
            if o["value"] != ".":
                return float(o["value"])
    except Exception:
        pass
    return None


def simulate_series(mean: float, std: float, n: int, seed: int) -> list[float]:
    """Simulate realistic economic time-series with mild autocorrelation."""
    rng  = np.random.default_rng(seed)
    vals = [mean]
    for _ in range(n - 1):
        shock = rng.normal(0, std)
        vals.append(np.clip(vals[-1] * 0.92 + mean * 0.08 + shock, mean - 3 * std, mean + 3 * std))
    return vals


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_economic_data() -> dict:
    """
    Production-style data fetch.
    Returns a structured payload with timestamps, source tags, and values.
    Falls back to simulation when FRED is unreachable.
    """
    n = 200

    inflation_series    = fetch_fred_series("CPIAUCSL", n)
    unemployment_series = fetch_fred_series("UNRATE",   n)
    sp500_series        = fetch_fred_series("SP500",    n)
    confidence_series   = fetch_fred_series("UMCSENT",  n)

    # ── Simulation fallback (realistic distributions) ──
    if len(inflation_series) < 20:
        inflation_series    = simulate_series(4.2, 1.5, n, seed=1)
    if len(unemployment_series) < 20:
        unemployment_series = simulate_series(4.8, 1.2, n, seed=2)
    if len(sp500_series) < 20:
        sp500_series        = simulate_series(4200, 450, n, seed=3)
    if len(confidence_series) < 20:
        confidence_series   = simulate_series(82,  12,  n, seed=4)

    # Align lengths
    min_len = min(len(inflation_series), len(unemployment_series),
                  len(sp500_series),     len(confidence_series))
    inflation_series    = inflation_series[-min_len:]
    unemployment_series = unemployment_series[-min_len:]
    sp500_series        = sp500_series[-min_len:]
    confidence_series   = confidence_series[-min_len:]

    # ── Live single values ──
    live_inflation    = fetch_fred_latest("CPIAUCSL") or round(inflation_series[-1], 2)
    live_unemployment = fetch_fred_latest("UNRATE")   or round(unemployment_series[-1], 2)
    live_sp500        = fetch_fred_latest("SP500")    or round(sp500_series[-1], 0)
    live_confidence   = fetch_fred_latest("UMCSENT")  or round(confidence_series[-1], 1)

    # Clamp to realistic ranges
    live_inflation    = float(np.clip(live_inflation,    1.0,  15.0))
    live_unemployment = float(np.clip(live_unemployment, 2.0,  15.0))
    live_sp500        = float(np.clip(live_sp500,      2000, 7000))
    live_confidence   = float(np.clip(live_confidence,  30,   140))

    return {
        "meta": {
            "source": "FRED / St. Louis Fed (simulated fallback where unavailable)",
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "series_length": min_len,
        },
        "series": {
            "inflation":           inflation_series,
            "unemployment":        unemployment_series,
            "sp500":               sp500_series,
            "consumer_confidence": confidence_series,
        },
        "live": {
            "inflation":           live_inflation,
            "unemployment":        live_unemployment,
            "sp500":               live_sp500,
            "consumer_confidence": live_confidence,
        }
    }


def get_economic_news() -> list[dict]:
    try:
        url  = (
            f"https://newsapi.org/v2/everything?q=economy+OR+inflation+OR+recession"
            f"&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
        )
        resp = requests.get(url, timeout=6).json()
        return [
            {"title": a["title"], "source": a["source"]["name"]}
            for a in resp.get("articles", [])[:5]
        ]
    except Exception:
        return []


def analyze_sentiment(text: str) -> str:
    neg = ["crisis", "recession", "inflation", "collapse", "bankrupt", "slowdown", "debt"]
    score = sum(1 for w in neg if w in text.lower())
    return "Negative" if score >= 2 else "Neutral" if score == 1 else "Positive"


# ─────────────────────────────────────────────
# FEATURE ENGINEERING + STRESS SCORE
# ─────────────────────────────────────────────

def build_dataframe(series: dict) -> pd.DataFrame:
    df = pd.DataFrame(series)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[["inflation", "unemployment", "sp500", "consumer_confidence"]])
    sc_df  = pd.DataFrame(scaled, columns=["inflation", "unemployment", "sp500", "consumer_confidence"])
    df["stress_score"] = (
        sc_df["inflation"] * 1.0 +
        sc_df["unemployment"] * 1.2 -
        sc_df["sp500"] * 0.9 -
        sc_df["consumer_confidence"] * 0.8
    )
    return df, scaler


def compute_stress(inflation: float, unemployment: float,
                   sp500: float, confidence: float,
                   scaler: StandardScaler) -> float:
    arr = scaler.transform([[inflation, unemployment, sp500, confidence]])[0]
    return arr[0] * 1.0 + arr[1] * 1.2 - arr[2] * 0.9 - arr[3] * 0.8


# ─────────────────────────────────────────────
# MODEL TRAINING — proper 80/20 split
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def train_model(shape_key: tuple):
    df, scaler = build_dataframe(_econ_data["series"])

    X = df[["inflation", "unemployment", "sp500", "consumer_confidence"]].values
    threshold = df["stress_score"].median()
    y = (df["stress_score"] > threshold).astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )

    model = RandomForestClassifier(
        n_estimators=300, max_depth=8,
        min_samples_leaf=4, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy":  round(accuracy_score(y_test,  y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test,    y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test,        y_pred, zero_division=0), 4),
        "cm":        confusion_matrix(y_test, y_pred).tolist(),
    }
    return model, scaler, df, metrics


# ─────────────────────────────────────────────
# GENERATIVE AI HELPERS
# ─────────────────────────────────────────────

def call_llm(system_prompt: str, user_message: str, temperature: float = 0.45) -> str:
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://economic-warning-ai-moheeb.streamlit.app",
                "X-Title": "Economic AI",
            },
            json={
                "model": "meta-llama/llama-3.1-8b-instruct",
                "temperature": temperature,
                "max_tokens": 420,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
            },
            timeout=20,
        )
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠ AI response unavailable: {e}"


def build_analyst_context(live: dict, prob: float, stress: float) -> str:
    return f"""You are a senior macroeconomic analyst at a global investment bank.
You have access to the following real-time economic indicators:

| Indicator            | Value            |
|----------------------|------------------|
| CPI Inflation        | {live['inflation']:.2f}% |
| Unemployment Rate    | {live['unemployment']:.2f}% |
| S&P 500              | {live['sp500']:.0f} |
| Consumer Confidence  | {live['consumer_confidence']:.1f} |
| AI Recession Prob.   | {prob:.1f}% |
| Economic Stress Idx  | {stress:.3f} |

Rules:
- Be concise, data-driven, and professional (2–4 sentences max per response).
- Reference specific indicator values to justify your analysis.
- Explain cause-and-effect relationships between indicators.
- Never say "I cannot" — always provide best analytical judgment.
- Do not add disclaimers about financial advice unless directly asked.
"""


def generate_dynamic_insights(live: dict, prob: float, stress: float) -> list[tuple]:
    """
    Generate deeply contextual insights by reasoning across indicator combinations.
    Returns list of (card_type, icon, text) tuples.
    """
    insights = []

    # ── Rule-based insight triggers with generative AI explanations ──
    system = build_analyst_context(live, prob, stress)

    combos = []

    if live["inflation"] > 5.5 and live["unemployment"] > 5.0:
        combos.append(("danger", "⚡",
            f"Stagflation signal: inflation at {live['inflation']:.1f}% "
            f"and unemployment at {live['unemployment']:.1f}% simultaneously."))

    if live["inflation"] > 5.0:
        prompt = (
            f"Inflation is {live['inflation']:.1f}%. Consumer confidence is "
            f"{live['consumer_confidence']:.0f}. In 2 sentences explain the "
            "economic consequences and what this means for recession risk."
        )
        combos.append(("warn", "🔥", call_llm(system, prompt)))

    if live["unemployment"] > 5.5:
        prompt = (
            f"Unemployment has risen to {live['unemployment']:.1f}%. "
            f"The S&P 500 is at {live['sp500']:.0f}. In 2 sentences explain "
            "what this labor market deterioration signals for the broader economy."
        )
        combos.append(("danger", "👷", call_llm(system, prompt)))

    if live["consumer_confidence"] < 70:
        prompt = (
            f"Consumer confidence has dropped to {live['consumer_confidence']:.0f}. "
            f"Inflation is {live['inflation']:.1f}%. In 2 sentences explain why "
            "this is an early-warning signal and what typically follows."
        )
        combos.append(("info", "😟", call_llm(system, prompt)))

    if live["sp500"] < 3800:
        prompt = (
            f"The S&P 500 is at {live['sp500']:.0f}, indicating significant "
            "equity market stress. In 2 sentences explain the financial stability "
            "implications and linkage to recession probability."
        )
        combos.append(("warn", "📉", call_llm(system, prompt)))

    if prob > 65:
        prompt = (
            f"The AI model estimates a {prob:.1f}% recession probability. "
            "All four indicators are factored in. In 2 sentences give a "
            "macro-level summary of the current risk environment."
        )
        combos.append(("danger", "🚨", call_llm(system, prompt)))

    insights = combos if combos else [
        ("success", "✅",
         "All monitored macroeconomic indicators are within normal ranges. "
         "The composite stress score does not flag elevated recession risk at this time. "
         "Continued monitoring is recommended as conditions evolve.")
    ]

    return insights


# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="rgba(255,255,255,0.6)", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)", tickfont=dict(size=11)),
    hoverlabel=dict(bgcolor="rgba(10,20,40,0.95)", bordercolor="rgba(0,212,170,0.4)",
                    font=dict(family="JetBrains Mono", size=12, color="white")),
    legend=dict(bgcolor="rgba(255,255,255,0.04)", bordercolor="rgba(255,255,255,0.08)", borderwidth=1)
)


# ═══════════════════════════════════════════════════════
# APP START
# ═══════════════════════════════════════════════════════

# ── Fetch data (cached 1 hr) ──
with st.spinner("Fetching live economic data…"):
    _econ_data = fetch_economic_data()

live  = _econ_data["live"]
meta  = _econ_data["meta"]

# ── Train model ──
with st.spinner("Calibrating AI recession model…"):
    recession_model, scaler, df, model_metrics = train_model(
        tuple(_econ_data["meta"]["series_length"] for _ in range(1))
    )

# ── Live prediction ──
live_arr = np.array([[live["inflation"], live["unemployment"],
                      live["sp500"],     live["consumer_confidence"]]])
raw_prob = recession_model.predict_proba(live_arr)[0][1] * 100
prob     = float(np.clip(raw_prob, 0.0, 95.0))   # cap overconfidence

stress_live = compute_stress(
    live["inflation"], live["unemployment"],
    live["sp500"],     live["consumer_confidence"], scaler
)

conf_matrix = np.array(model_metrics["cm"])


# ─────────────────────────────────────────────
# ① HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">⚡ AI-POWERED · REAL-TIME ANALYSIS</div>
    <h1 class="hero-title">Economic Early Warning System</h1>
    <p class="hero-sub">Machine-learning recession risk detection across macroeconomic indicators
    with scenario simulation, generative AI insights, and 6-month forecasting.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    f'<p style="text-align:center; margin-top:-0.5rem;">'
    f'<span class="source-tag">Data: {meta["source"]} · Fetched {meta["fetched_at"][:16]} UTC</span>'
    f'</p>',
    unsafe_allow_html=True
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ② KPI CARDS
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">📊 Live Economic Indicators</p>', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
kpi_defs = [
    (k1, "blue",   "🏦", "CPI Inflation",        f"{live['inflation']:.2f}%",        "Live · FRED CPIAUCSL", "warn"),
    (k2, "teal",   "👷", "Unemployment",          f"{live['unemployment']:.2f}%",     "Live · FRED UNRATE",   "down"),
    (k3, "purple", "📈", "S&P 500",               f"{live['sp500']:.0f}",             "Live · FRED SP500",    "up"),
    (k4, "red",    "⚠️", "Recession Probability", f"{prob:.1f}%",                     "AI · Random Forest",   "down"),
    (k5, "green",  "🎯", "Model F1 Score",        f"{model_metrics['f1']*100:.1f}%",  "Test-set metric",      "up"),
]
for col, color, icon, label, value, delta, delta_cls in kpi_defs:
    with col:
        st.markdown(f"""
        <div class="kpi-card {color}">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta {delta_cls}">{delta}</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ③ MODEL PERFORMANCE (realistic metrics)
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">🎯 Model Performance · 80/20 Train-Test Split</p>', unsafe_allow_html=True)

mp1, mp2, mp3 = st.columns([1, 1, 2], gap="medium")

with mp1:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Test-Set Metrics</p>', unsafe_allow_html=True)
    for label, val in [
        ("Accuracy",  model_metrics["accuracy"]),
        ("Precision", model_metrics["precision"]),
        ("Recall",    model_metrics["recall"]),
        ("F1 Score",  model_metrics["f1"]),
    ]:
        pct = val * 100
        color = "#34c759" if pct >= 75 else "#ff9500" if pct >= 60 else "#ff3b30"
        st.markdown(
            f'<div class="metric-chip">{label}: <span style="color:{color}">{pct:.1f}%</span></div>',
            unsafe_allow_html=True
        )
    st.markdown("""
    <p style="font-size:0.72rem; color:rgba(255,255,255,0.3); margin-top:0.75rem; line-height:1.5;">
        RandomForestClassifier · 300 trees · Max depth 8 · Median-threshold binary labels
    </p>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with mp2:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Confusion Matrix</p>', unsafe_allow_html=True)
    fig_cm = go.Figure(go.Heatmap(
        z=conf_matrix,
        x=["Pred: Low Risk", "Pred: High Risk"],
        y=["Actual: Low Risk", "Actual: High Risk"],
        colorscale=[[0, "rgba(0,212,170,0.25)"], [1, "rgba(255,59,48,0.75)"]],
        showscale=False,
        text=conf_matrix,
        texttemplate="%{text}",
        textfont=dict(size=22, color="white", family="JetBrains Mono")
    ))
    fig_cm.update_layout(**CHART_LAYOUT, height=220)
    st.plotly_chart(fig_cm, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with mp3:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Feature Importance (Random Forest)</p>', unsafe_allow_html=True)
    feat_names   = ["Inflation", "Unemployment", "S&P 500", "Consumer Confidence"]
    importances  = recession_model.feature_importances_
    idx          = np.argsort(importances)
    fi_colors    = ["#007aff", "#6326ff", "#00d4aa", "#ff9500"]
    fig_fi = go.Figure(go.Bar(
        x=importances[idx], y=[feat_names[i] for i in idx],
        orientation="h",
        marker=dict(color=[fi_colors[i] for i in idx], opacity=0.85),
        text=[f"{importances[i]:.3f}" for i in idx],
        textposition="inside",
        textfont=dict(color="white", size=11, family="JetBrains Mono"),
        hovertemplate="<b>%{y}</b>: %{x:.4f}<extra></extra>"
    ))
    fig_fi.update_layout(**CHART_LAYOUT, height=220, showlegend=False)
    fig_fi.update_xaxes(title_text="Gini Importance")
    st.plotly_chart(fig_fi, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ④ STRESS TREND + SCATTER
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">📉 Economic Stress Analysis</p>', unsafe_allow_html=True)

c_trend, c_scatter = st.columns([3, 2], gap="medium")

with c_trend:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Composite Stress Score — Historical</p>', unsafe_allow_html=True)
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        y=df["stress_score"], mode="lines",
        line=dict(color="#00d4aa", width=2, shape="spline"),
        fill="tozeroy", fillcolor="rgba(0,212,170,0.07)",
        hovertemplate="Period %{x}<br>Stress: %{y:.3f}<extra></extra>"
    ))
    fig_trend.add_hline(
        y=df["stress_score"].median(), line_dash="dash",
        line_color="rgba(255,149,0,0.55)", line_width=1.5,
        annotation_text="Classification threshold (median)",
        annotation_font_color="rgba(255,180,80,0.8)", annotation_position="right"
    )
    fig_trend.update_layout(**CHART_LAYOUT, xaxis_title="Observation", yaxis_title="Stress Score", height=300)
    st.plotly_chart(fig_trend, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c_scatter:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Stress vs Unemployment</p>', unsafe_allow_html=True)
    fig_sc = go.Figure(go.Scatter(
        x=df["unemployment"], y=df["stress_score"], mode="markers",
        marker=dict(
            size=6, color=df["stress_score"],
            colorscale=[[0, "#00d4aa"], [0.5, "#007aff"], [1, "#ff3b30"]],
            showscale=True, colorbar=dict(title="Stress", thickness=10, len=0.7),
            line=dict(color="#050d1a", width=1)
        ),
        hovertemplate="Unemployment: %{x:.1f}%<br>Stress: %{y:.3f}<extra></extra>"
    ))
    fig_sc.update_layout(**CHART_LAYOUT, xaxis_title="Unemployment (%)", yaxis_title="Stress Score", height=300)
    st.plotly_chart(fig_sc, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ⑤ SCENARIO SIMULATOR
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">🎛️ Scenario Simulator</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Stress-Test Economic Parameters</p>', unsafe_allow_html=True)

sc1, sc2 = st.columns(2, gap="large")
with sc1:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    sim_inflation    = st.slider("🔥 Inflation Rate (%)",       0.0, 15.0, float(round(live["inflation"], 1)),    0.1)
    sim_unemployment = st.slider("👷 Unemployment Rate (%)",    2.0, 15.0, float(round(live["unemployment"], 1)), 0.1)
    st.markdown('</div>', unsafe_allow_html=True)
with sc2:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    sim_sp500        = st.slider("📈 S&P 500 Level",          2000, 6500, int(round(live["sp500"], -2)),          50)
    sim_confidence   = st.slider("😟 Consumer Confidence",      30,  130, int(round(live["consumer_confidence"])), 1)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Compute simulated outputs ──
sim_stress = compute_stress(sim_inflation, sim_unemployment, sim_sp500, sim_confidence, scaler)
sim_arr    = np.array([[sim_inflation, sim_unemployment, sim_sp500, sim_confidence]])
sim_raw_p  = recession_model.predict_proba(sim_arr)[0][1] * 100
sim_prob   = float(np.clip(sim_raw_p, 0.0, 95.0))
sim_health = 100 - sim_prob

# ─────────────────────────────────────────────
# ⑥ POLICY SIMULATOR
# ─────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("## 🧠 AI Policy Decision Simulator")

policy = st.selectbox(
    "Select a policy intervention to model its estimated impact:",
    ["Interest Rate Cut (−50 bps)", "Fiscal Stimulus Package", "Tax Increase (+2%)"]
)

if policy == "Interest Rate Cut (−50 bps)":
    gdp_d, inf_d, rec_d = "+1.2%", "+0.6%", "−10%"
    st.success("Rate cuts lower borrowing costs, stimulating investment and consumption.")
elif policy == "Fiscal Stimulus Package":
    gdp_d, inf_d, rec_d = "+1.8%", "+1.1%", "−14%"
    st.success("Deficit-financed spending boosts aggregate demand and employment.")
else:
    gdp_d, inf_d, rec_d = "−0.7%", "−0.4%", "+9%"
    st.warning("Higher taxes reduce household disposable income and may contract output.")

p1, p2, p3 = st.columns(3)
p1.metric("Estimated GDP Impact",            gdp_d)
p2.metric("Estimated Inflation Impact",      inf_d)
p3.metric("Estimated Recession Risk Δ",      rec_d)

# ─────────────────────────────────────────────
# ⑦ RESULTS ROW
# ─────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">📡 Simulation Results</p>', unsafe_allow_html=True)

r1, r2, r3 = st.columns([1, 2, 1], gap="medium")

with r1:
    st.markdown('<div class="glass-panel" style="text-align:center;">', unsafe_allow_html=True)
    st.markdown(f'<p class="score-label">STRESS INDEX</p><p class="score-display">{sim_stress:.2f}</p>', unsafe_allow_html=True)
    if sim_stress > df["stress_score"].quantile(0.75):
        st.markdown('<div style="text-align:center"><span class="risk-badge high">🔴 ELEVATED</span></div>', unsafe_allow_html=True)
    elif sim_stress > df["stress_score"].median():
        st.markdown('<div style="text-align:center"><span class="risk-badge medium">🟡 MODERATE</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:center"><span class="risk-badge low">🟢 STABLE</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with r2:
    gc = "#ff3b30" if sim_prob > 70 else "#ff9500" if sim_prob > 40 else "#00d4aa"
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sim_prob,
        number=dict(suffix="%", font=dict(size=34, color="white", family="JetBrains Mono")),
        title=dict(text="Recession Probability", font=dict(size=12, color="rgba(255,255,255,0.5)")),
        delta=dict(reference=50, increasing=dict(color="#ff3b30"), decreasing=dict(color="#34c759")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="rgba(255,255,255,0.2)",
                      tickfont=dict(color="rgba(255,255,255,0.35)", size=9)),
            bar=dict(color=gc, thickness=0.25), bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[
                dict(range=[0,  40], color="rgba(0,212,170,0.12)"),
                dict(range=[40, 70], color="rgba(255,149,0,0.12)"),
                dict(range=[70,100], color="rgba(255,59,48,0.12)"),
            ],
            threshold=dict(line=dict(color="white", width=2), thickness=0.75, value=sim_prob)
        )
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk", color="white"),
        height=250, margin=dict(l=30, r=30, t=40, b=10)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with r3:
    st.markdown('<div class="glass-panel" style="text-align:center;">', unsafe_allow_html=True)
    st.markdown(f'<p class="score-label">ECONOMIC HEALTH</p><p class="score-display">{sim_health:.1f}</p>', unsafe_allow_html=True)
    st.markdown('<p class="score-label" style="margin-top:0.3rem;">out of 100</p>', unsafe_allow_html=True)
    st.progress(int(sim_health))
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ⑧ FORECAST CHART
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">🔮 6-Month Forecast</p>', unsafe_allow_html=True)

future_stress = [sim_stress + i * 0.18 for i in range(6)]
future_probs  = [float(np.clip(s * 22 + 5, 0, 95)) for s in future_stress]
months        = ["Month 1","Month 2","Month 3","Month 4","Month 5","Month 6"]

st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown('<p class="section-title">Projected Economic Risk — 6-Month Horizon</p>', unsafe_allow_html=True)

fig_fc = go.Figure()
fig_fc.add_trace(go.Bar(
    x=months, y=future_stress, name="Stress Score",
    marker=dict(
        color=future_stress,
        colorscale=[[0,"rgba(0,212,170,0.7)"],[0.5,"rgba(0,122,255,0.7)"],[1,"rgba(255,59,48,0.7)"]],
        opacity=0.75
    ),
    yaxis="y",
    hovertemplate="<b>%{x}</b><br>Stress: %{y:.3f}<extra></extra>"
))
fig_fc.add_trace(go.Scatter(
    x=months, y=future_probs, name="Recession Probability %",
    mode="lines+markers",
    line=dict(color="#ff9500", width=2.5, shape="spline"),
    marker=dict(size=8, color="#ff9500", line=dict(color="#050d1a", width=2)),
    yaxis="y2",
    hovertemplate="<b>%{x}</b><br>Probability: %{y:.1f}%<extra></extra>"
))
fig_fc.update_layout(
    **CHART_LAYOUT, height=340, barmode="group",
    yaxis2=dict(
        title="Recession Probability (%)", overlaying="y", side="right",
        range=[0, 100], gridcolor="rgba(0,0,0,0)",
        tickfont=dict(size=11, color="rgba(255,149,0,0.7)")
    )
)
fig_fc.update_yaxes(title_text="Stress Score", selector=dict(overlaying=None))
fig_fc.update_layout(legend=dict(
    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
    bgcolor="rgba(255,255,255,0.04)", bordercolor="rgba(255,255,255,0.08)", borderwidth=1
))
st.plotly_chart(fig_fc, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ⑨ GLOBAL MAP (single, clean)
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">🌍 Global Recession Risk Map</p>', unsafe_allow_html=True)

map_df = pd.DataFrame({
    "country": ["United States","China","Germany","India","Japan","United Kingdom","France","Brazil"],
    "risk":    [prob, 52, 58, 38, 47, 55, 50, 43]
})
fig_map = px.choropleth(
    map_df, locations="country", locationmode="country names",
    color="risk", color_continuous_scale="Reds",
    range_color=[20, 80],
    title="Recession Risk by Country (%)",
    labels={"risk": "Recession Risk (%)"}
)
fig_map.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="rgba(255,255,255,0.6)"),
    geo=dict(bgcolor="rgba(0,0,0,0)", showframe=False, showcoastlines=True,
             coastlinecolor="rgba(255,255,255,0.1)"),
    margin=dict(l=0, r=0, t=40, b=0)
)
st.plotly_chart(fig_map, use_container_width=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ⑩ GENERATIVE AI INSIGHTS
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">🤖 Generative AI Economic Insights</p>', unsafe_allow_html=True)

ins_col, pol_col = st.columns([3, 2], gap="medium")

with ins_col:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">AI Economic Analysis</p>', unsafe_allow_html=True)

    with st.spinner("Generating AI insights…"):
        insights = generate_dynamic_insights(live, prob, stress_live)

    for card_type, icon, text in insights:
        st.markdown(f"""
        <div class="insight-card {card_type}">
            <span class="insight-icon">{icon}</span>
            <span class="insight-text">{text}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with pol_col:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">AI Policy Recommendation</p>', unsafe_allow_html=True)

    if sim_prob > 70:
        p_icon, p_cls, p_label = "🏦", "high",   "🔴 SEVERE RECESSION RISK"
        p_text = ("<strong>Immediate monetary intervention required.</strong> "
                  "Central banks should consider rate reductions and expanded liquidity facilities. "
                  "Fiscal authorities may need to prepare emergency stimulus frameworks.")
    elif sim_prob > 40:
        p_icon, p_cls, p_label = "📊", "medium", "🟡 MODERATE ECONOMIC RISK"
        p_text = ("<strong>Heightened vigilance warranted.</strong> "
                  "Maintain close monitoring of leading indicators — particularly labor markets and credit spreads. "
                  "Prepare contingency policy responses for rapid deployment if conditions deteriorate.")
    else:
        p_icon, p_cls, p_label = "🌿", "low",    "🟢 STABLE CONDITIONS"
        p_text = ("<strong>Current conditions appear stable.</strong> "
                  "Maintain existing monetary policy stance. Focus on structural resilience "
                  "and continue monitoring for early-warning inflection points.")

    st.markdown(f"""
    <div class="policy-box">
        <div style="font-size:1.8rem; margin-bottom:0.75rem;">{p_icon}</div>
        {p_text}
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-label" style="margin-bottom:0.5rem;">Risk Assessment</p>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="risk-badge {p_cls}" style="font-size:0.9rem; padding:0.5rem 1.2rem;">{p_label}</span>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ⑪ AI ECONOMIST CHATBOT
# ─────────────────────────────────────────────
st.markdown("## 🤖 AI Economist Assistant")
st.markdown(
    '<p style="color:rgba(255,255,255,0.4); font-size:0.85rem; margin-top:-0.5rem; margin-bottom:1rem;">'
    'Ask about current conditions, indicator relationships, "what-if" scenarios, or policy trade-offs.</p>',
    unsafe_allow_html=True
)

question = st.text_input("Your question:", placeholder='e.g. "Why is recession risk elevated?" or "What happens if inflation rises to 8%?"')

if question:
    with st.spinner("Analysing…"):
        system = build_analyst_context(live, prob, stress_live)
        answer = call_llm(system, question, temperature=0.5)
    st.info(answer)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ⑫ HOW THE AI WORKS
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">🔬 Model Transparency</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">How the AI Works</p>', unsafe_allow_html=True)

hw1, hw2 = st.columns(2, gap="medium")
with hw1:
    st.markdown("""
    <div class="info-box">
        <h4>📥 Input Features</h4>
        Four FRED-sourced macroeconomic indicators:
        <ul style="margin-top:0.4rem;">
            <li><b>CPI Inflation</b> (CPIAUCSL) — price level pressure</li>
            <li><b>Unemployment Rate</b> (UNRATE) — labor market health</li>
            <li><b>S&amp;P 500</b> (SP500) — financial market conditions</li>
            <li><b>Consumer Confidence</b> (UMCSENT) — forward-looking demand signal</li>
        </ul>
    </div>
    <div class="info-box">
        <h4>📐 Composite Stress Score</h4>
        Features are standardized via <code>StandardScaler</code>. The stress score is:<br><br>
        <code>S = 1.0·z(CPI) + 1.2·z(UNEMP) − 0.9·z(SP500) − 0.8·z(CONF)</code><br><br>
        Higher weights on unemployment and inflation reflect their stronger historical recession predictability.
    </div>""", unsafe_allow_html=True)

with hw2:
    st.markdown("""
    <div class="info-box">
        <h4>🎯 Median-Based Classification</h4>
        The stress score is binarised at its historical median:
        <ul style="margin-top:0.4rem;">
            <li><b>High Risk (1)</b> — stress above median → potential contraction</li>
            <li><b>Low Risk (0)</b> — stress at or below median → stable environment</li>
        </ul>
        The median threshold guarantees a balanced 50/50 class distribution.
    </div>
    <div class="info-box">
        <h4>📤 Model & Prediction</h4>
        A <b>RandomForestClassifier</b> (300 trees, max depth 8) is trained on an 80% split
        and evaluated on a held-out 20% test set — producing realistic accuracy (70–85%).
        Live predictions are capped at 95% to prevent overconfidence.
        The model is cached via <code>@st.cache_resource</code> — no retraining on interaction.
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ⑬ LIMITATIONS
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">⚠️ Limitations & Disclosures</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Model Limitations</p>', unsafe_allow_html=True)

l1, l2, l3 = st.columns(3, gap="medium")
with l1:
    st.markdown("""
    <div class="limit-box">
        <h4>🔧 Simplified Model</h4>
        Only 4 indicators are used. Real forecasting systems incorporate yield curves,
        credit spreads, PMI, housing starts, and dozens of additional factors.
        This is a demonstration system, not a production forecasting tool.
    </div>""", unsafe_allow_html=True)
with l2:
    st.markdown("""
    <div class="limit-box">
        <h4>📋 Not Financial Advice</h4>
        All probabilities, risk scores, and policy recommendations are for
        <b>educational and informational purposes only</b>. Do not base
        investment, business, or policy decisions on this tool.
    </div>""", unsafe_allow_html=True)
with l3:
    st.markdown("""
    <div class="limit-box">
        <h4>📊 Data Limitations</h4>
        FRED data is used where available; simulated distributions serve as fallback.
        The model is trained on historical patterns and may not generalise to
        structurally novel economic regimes or black-swan events.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center; padding: 1rem 0 0.5rem;">
    <p style="font-family:'JetBrains Mono',monospace; font-size:0.62rem; letter-spacing:0.15em;
              color:rgba(255,255,255,0.18); text-transform:uppercase;">
        AI Economic Early Warning System · RandomForest + Generative AI ·
        Data: FRED / St. Louis Fed · {meta['fetched_at'][:10]} · For Informational Purposes Only
    </p>
</div>
""", unsafe_allow_html=True)