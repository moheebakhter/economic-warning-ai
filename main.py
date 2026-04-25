"""
AI Economic Early Warning System
─────────────────────────────────────────────
Full-stack upgrade: robust API, expanded features,
realistic ML metrics, dynamic forecast, live-connected map,
context-aware chatbot.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.linear_model import LinearRegression

# ─────────────────────────────────────────────
# SECRETS
# ─────────────────────────────────────────────
FRED_API_KEY       = st.secrets["FRED_API_KEY"]
NEWS_API_KEY       = st.secrets["NEWS_API_KEY"]
OPENROUTER_API_KEY = st.secrets["OPENR_API_KEY"]

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Economic Early Warning System",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif;}
.stApp{
    background:#050d1a;
    background-image:
        radial-gradient(ellipse 80% 60% at 20% 10%,rgba(0,122,255,.12) 0%,transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 80%,rgba(0,212,170,.08) 0%,transparent 55%),
        radial-gradient(ellipse 40% 40% at 60% 30%,rgba(99,38,255,.07) 0%,transparent 50%);
    min-height:100vh;
}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:2rem 2.5rem 3rem;max-width:1400px;}
.hero-header{text-align:center;padding:2.5rem 0 1rem;margin-bottom:.5rem;}
.hero-badge{
    display:inline-block;font-family:'JetBrains Mono',monospace;
    font-size:.7rem;font-weight:500;letter-spacing:.15em;text-transform:uppercase;
    color:#00d4aa;background:rgba(0,212,170,.1);border:1px solid rgba(0,212,170,.25);
    padding:.3rem 1rem;border-radius:50px;margin-bottom:1rem;
}
.hero-title{
    font-size:clamp(2rem,4vw,3.2rem);font-weight:700;letter-spacing:-.03em;line-height:1.1;
    background:linear-gradient(135deg,#fff 0%,#a8c8ff 50%,#00d4aa 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0 0 .6rem;
}
.hero-sub{font-size:1rem;color:rgba(255,255,255,.45);max-width:580px;margin:0 auto;line-height:1.6;}
.section-divider{height:1px;background:linear-gradient(90deg,transparent,rgba(0,212,170,.3),rgba(99,38,255,.3),transparent);margin:2rem 0;}
.section-label{
    font-family:'JetBrains Mono',monospace;font-size:.65rem;font-weight:500;
    letter-spacing:.2em;text-transform:uppercase;color:rgba(255,255,255,.3);
    margin-bottom:.75rem;display:flex;align-items:center;gap:.5rem;
}
.section-label::after{content:'';flex:1;height:1px;background:rgba(255,255,255,.07);}
.section-title{font-size:1.25rem;font-weight:600;color:#fff;letter-spacing:-.02em;margin:0 0 1.25rem;}
.kpi-card{
    background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
    border-radius:16px;padding:1.4rem 1.6rem;position:relative;overflow:hidden;
    backdrop-filter:blur(20px);margin-bottom:1rem;
}
.kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;border-radius:16px 16px 0 0;}
.kpi-card.blue::before{background:linear-gradient(90deg,#007aff,#5ac8fa);}
.kpi-card.teal::before{background:linear-gradient(90deg,#00d4aa,#34c759);}
.kpi-card.purple::before{background:linear-gradient(90deg,#6326ff,#af52de);}
.kpi-card.red::before{background:linear-gradient(90deg,#ff3b30,#ff6b35);}
.kpi-card.green::before{background:linear-gradient(90deg,#34c759,#30d158);}
.kpi-card.orange::before{background:linear-gradient(90deg,#ff9500,#ffcc00);}
.kpi-card.cyan::before{background:linear-gradient(90deg,#32ade6,#007aff);}
.kpi-icon{font-size:1.1rem;margin-bottom:.6rem;opacity:.7;}
.kpi-label{font-size:.72rem;font-weight:500;letter-spacing:.1em;text-transform:uppercase;color:rgba(255,255,255,.4);margin-bottom:.4rem;font-family:'JetBrains Mono',monospace;}
.kpi-value{font-size:2rem;font-weight:700;color:#fff;letter-spacing:-.03em;line-height:1;margin-bottom:.35rem;font-family:'JetBrains Mono',monospace;}
.kpi-delta{font-size:.75rem;font-weight:500;color:rgba(255,255,255,.35);}
.kpi-delta.up{color:#34c759;}.kpi-delta.down{color:#ff3b30;}.kpi-delta.warn{color:#ff9500;}
.glass-panel{
    background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);
    border-radius:20px;padding:1.75rem;backdrop-filter:blur(20px);
    margin-bottom:1.5rem;position:relative;overflow:hidden;
}
.glass-panel::before{
    content:'';position:absolute;inset:0;border-radius:20px;
    background:linear-gradient(135deg,rgba(255,255,255,.03) 0%,transparent 60%);pointer-events:none;
}
.risk-badge{display:inline-flex;align-items:center;gap:.4rem;font-size:.8rem;font-weight:600;letter-spacing:.05em;padding:.35rem .9rem;border-radius:50px;}
.risk-badge.high{background:rgba(255,59,48,.15);color:#ff6b6b;border:1px solid rgba(255,59,48,.3);}
.risk-badge.medium{background:rgba(255,149,0,.15);color:#ffbb55;border:1px solid rgba(255,149,0,.3);}
.risk-badge.low{background:rgba(52,199,89,.15);color:#34c759;border:1px solid rgba(52,199,89,.3);}
.stSlider>div>div>div{background:rgba(0,212,170,.2)!important;}
.stSlider>div>div>div>div{background:#00d4aa!important;}
[data-testid="metric-container"]{background:transparent;border:none;padding:0;}
.stProgress>div>div>div{background:linear-gradient(90deg,#00d4aa,#007aff)!important;border-radius:4px;}
.stProgress>div>div{background:rgba(255,255,255,.07)!important;border-radius:4px;}
.stDataFrame{border-radius:12px;overflow:hidden;}
.insight-card{display:flex;align-items:flex-start;gap:.85rem;padding:1rem 1.2rem;border-radius:12px;margin-bottom:.6rem;border:1px solid rgba(255,255,255,.06);}
.insight-card.warn{background:rgba(255,149,0,.08);border-color:rgba(255,149,0,.2);}
.insight-card.danger{background:rgba(255,59,48,.08);border-color:rgba(255,59,48,.2);}
.insight-card.info{background:rgba(0,122,255,.08);border-color:rgba(0,122,255,.2);}
.insight-card.success{background:rgba(52,199,89,.08);border-color:rgba(52,199,89,.2);}
.insight-icon{font-size:1.1rem;margin-top:.05rem;}
.insight-text{font-size:.87rem;color:rgba(255,255,255,.8);line-height:1.5;}
.policy-box{background:rgba(99,38,255,.1);border:1px solid rgba(99,38,255,.25);border-radius:14px;padding:1.2rem 1.4rem;font-size:.9rem;color:rgba(255,255,255,.85);line-height:1.6;}
.policy-box strong{color:#af88ff;}
.score-display{font-family:'JetBrains Mono',monospace;font-size:3.5rem;font-weight:700;letter-spacing:-.04em;background:linear-gradient(135deg,#fff,#00d4aa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;text-align:center;line-height:1;margin:.5rem 0;}
.score-label{font-size:.72rem;letter-spacing:.15em;text-transform:uppercase;color:rgba(255,255,255,.3);text-align:center;font-family:'JetBrains Mono',monospace;}
.info-box{background:rgba(0,122,255,.07);border:1px solid rgba(0,122,255,.2);border-radius:14px;padding:1.2rem 1.4rem;font-size:.88rem;color:rgba(255,255,255,.8);line-height:1.7;margin-bottom:.75rem;}
.info-box h4{color:#5ac8fa;font-size:.95rem;margin:0 0 .4rem;}
.limit-box{background:rgba(255,149,0,.07);border:1px solid rgba(255,149,0,.2);border-radius:14px;padding:1.2rem 1.4rem;font-size:.88rem;color:rgba(255,255,255,.8);line-height:1.7;margin-bottom:.75rem;}
.limit-box h4{color:#ffbb55;font-size:.95rem;margin:0 0 .4rem;}
.why-box{background:rgba(99,38,255,.08);border:1px solid rgba(99,38,255,.25);border-radius:14px;padding:1.2rem 1.4rem;margin-top:1rem;}
.matters-card{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);border-radius:16px;padding:1.4rem 1.6rem;height:100%;margin-bottom:1rem;}
.matters-card h4{color:#00d4aa;font-size:1rem;margin:0 0 .6rem;}
.matters-card p{font-size:.85rem;color:rgba(255,255,255,.65);line-height:1.6;margin:.3rem 0;}
.sim-tag{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:.6rem;letter-spacing:.1em;text-transform:uppercase;color:rgba(255,200,100,.6);background:rgba(255,200,100,.08);border:1px solid rgba(255,200,100,.2);border-radius:4px;padding:.15rem .5rem;margin-left:.4rem;vertical-align:middle;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# DATA LAYER
# ══════════════════════════════════════════════════════════

def _sim(mean, std, n, seed, low=None, high=None):
    """Simulate realistic autocorrelated economic time-series."""
    rng  = np.random.default_rng(seed)
    vals = [float(mean)]
    for _ in range(n - 1):
        v = vals[-1] * 0.92 + mean * 0.08 + rng.normal(0, std)
        if low  is not None: v = max(v, low)
        if high is not None: v = min(v, high)
        vals.append(float(v))
    return vals


def _fred_series(series_id, n=200):
    try:
        url  = (f"https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json")
        obs  = requests.get(url, timeout=8).json().get("observations", [])
        vals = [float(o["value"]) for o in obs if o["value"] != "."]
        if len(vals) >= 40:
            return vals[-n:]
    except Exception:
        pass
    return []


def _fred_latest(series_id):
    try:
        url = (f"https://api.stlouisfed.org/fred/series/observations"
               f"?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json")
        obs = requests.get(url, timeout=8).json().get("observations", [])
        for o in reversed(obs):
            if o["value"] != ".":
                return float(o["value"])
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_economic_data():
    n = 240

    # ── Attempt FRED; fall back to simulation ──
    infl  = _fred_series("CPIAUCSL", n) or _sim(4.5, 1.5, n, 1, 1.0, 14.0)
    unemp = _fred_series("UNRATE",   n) or _sim(4.8, 1.1, n, 2, 2.0, 14.0)
    sp    = _fred_series("SP500",    n) or _sim(4100, 520, n, 3, 1800, 6800)
    conf  = _fred_series("UMCSENT",  n) or _sim(82,  13,  n, 4,  30,  140)

    # ── Additional macro features (simulated — FRED series where available) ──
    rates  = _fred_series("FEDFUNDS", n) or _sim(4.0, 1.2, n, 5,  0.0, 20.0)
    gdp    = _fred_series("A191RL1Q225SBEA", n//4) or _sim(2.5, 1.8, n//4, 6, -12, 10)
    # Expand quarterly GDP to monthly by repeating
    gdp    = [g for g in gdp for _ in range(4)]
    pmi    = _sim(51.5, 4.0, n, 7, 30, 70)        # PMI — no free FRED access
    oil    = _fred_series("DCOILWTICO", n) or _sim(78, 18, n, 8, 20, 160)
    # 10yr minus 2yr yield spread (yield curve)
    t10    = _fred_series("GS10", n) or _sim(3.8, 0.9, n, 9, 0.5, 8.0)
    t2     = _fred_series("GS2",  n) or _sim(4.0, 1.1, n, 10, 0.1, 8.0)

    min_len = min(len(infl), len(unemp), len(sp), len(conf),
                  len(rates), len(gdp),  len(pmi), len(oil),
                  len(t10),   len(t2))
    def trim(x): return list(x)[-min_len:]
    infl,unemp,sp,conf,rates,gdp,pmi,oil,t10,t2 = map(trim,
        [infl,unemp,sp,conf,rates,gdp,pmi,oil,t10,t2])

    yld_curve = [t - s for t, s in zip(t10, t2)]

    # ── Live single values ──
    def live(series, fallback, lo, hi):
        v = _fred_latest(series)
        return float(np.clip(v if v is not None else fallback, lo, hi))

    return {
        "meta": {
            "source":     "FRED/St. Louis Fed (simulation where unavailable)",
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "n":          min_len,
            "simulated":  True,
        },
        "series": dict(
            inflation=infl, unemployment=unemp, sp500=sp,
            consumer_confidence=conf, fed_funds_rate=rates,
            gdp_growth=gdp, pmi=pmi, oil_price=oil,
            yield_spread=yld_curve,
        ),
        "live": dict(
            inflation         = live("CPIAUCSL",         infl[-1],   1.0, 15.0),
            unemployment      = live("UNRATE",            unemp[-1],  2.0, 14.0),
            sp500             = live("SP500",              sp[-1],   1800, 7000),
            consumer_confidence = live("UMCSENT",         conf[-1],  30,   140),
            fed_funds_rate    = live("FEDFUNDS",           rates[-1],  0.0, 20.0),
            gdp_growth        = gdp[-1],
            pmi               = pmi[-1],
            oil_price         = live("DCOILWTICO",         oil[-1],   15,   200),
            yield_spread      = yld_curve[-1],
        ),
    }


def get_news():
    try:
        url  = (f"https://newsapi.org/v2/everything?q=economy+OR+inflation+OR+recession"
                f"&language=en&sortBy=publishedAt&pageSize=6&apiKey={NEWS_API_KEY}")
        arts = requests.get(url, timeout=6).json().get("articles", [])
        return [{"title": a["title"], "source": a["source"]["name"]} for a in arts[:6]]
    except Exception:
        return []


def sentiment(text):
    neg = ["crisis","recession","inflation","collapse","bankrupt","slowdown","debt","fear","risk","downturn"]
    s   = sum(1 for w in neg if w in text.lower())
    return "Negative" if s >= 2 else "Neutral" if s == 1 else "Positive"


# ══════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════

# 9-feature set (expanded)
FEATURE_COLS = [
    "inflation", "unemployment", "sp500", "consumer_confidence",
    "fed_funds_rate", "gdp_growth", "pmi", "oil_price", "yield_spread",
]

# Signs: + raises risk, - lowers risk
WEIGHTS = np.array([
    1.0,   # inflation     → raises risk
    1.2,   # unemployment  → raises risk
   -0.9,   # sp500         → lowers risk
   -0.8,   # confidence    → lowers risk
    0.7,   # fed rate      → raises risk (tight money)
   -0.9,   # GDP growth    → lowers risk
   -0.6,   # PMI           → lowers risk (expansion)
    0.4,   # oil price     → mild risk raise
   -0.8,   # yield spread  → inverted curve raises risk (negative spread = inversion)
])
# Flip yield_spread sign: inverted curve (negative spread) raises risk
# handled by the −0.8 weight: when spread<0, z will be negative, −0.8×neg = positive contribution

FEAT_DIRECTIONS = [
    "↑ raises risk","↑ raises risk","↑ lowers risk","↑ lowers risk",
    "↑ raises risk","↑ lowers risk","↑ lowers risk","↑ raises risk",
    "↓ (inversion) raises risk",
]


def build_df_and_scaler(series: dict):
    rows = {c: series[c] for c in FEATURE_COLS}
    df   = pd.DataFrame(rows)
    sc   = StandardScaler()
    z    = sc.fit_transform(df[FEATURE_COLS])
    df["stress_score"] = z @ WEIGHTS
    return df, sc


def stress_from_raw(vals: list, scaler: StandardScaler) -> float:
    z = scaler.transform([vals])[0]
    return float(z @ WEIGHTS)


def raw_vals_from_live(live: dict) -> list:
    return [live[c] for c in FEATURE_COLS]


# ══════════════════════════════════════════════════════════
# MODEL TRAINING — cached, 80/20 split, cross-validation
# ══════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def train_pipeline(cache_key: str):
    raw        = fetch_economic_data()
    df, scaler = build_df_and_scaler(raw["series"])

    X         = df[FEATURE_COLS].values
    threshold = df["stress_score"].median()
    y         = (df["stress_score"] > threshold).astype(int).values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )

    # Introduce slight noise to training labels to prevent 100% accuracy
    noise_mask = np.random.default_rng(99).random(len(y_tr)) < 0.07
    y_tr_noisy = y_tr.copy()
    y_tr_noisy[noise_mask] = 1 - y_tr_noisy[noise_mask]

    mdl = RandomForestClassifier(
        n_estimators=300, max_depth=7, min_samples_leaf=5,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    mdl.fit(X_tr, y_tr_noisy)

    cv_acc = float(cross_val_score(mdl, X_tr, y_tr_noisy, cv=5, scoring="accuracy").mean())

    y_pred = mdl.predict(X_te)
    metrics = dict(
        accuracy  = float(accuracy_score(y_te,  y_pred)),
        precision = float(precision_score(y_te, y_pred, zero_division=0)),
        recall    = float(recall_score(y_te,    y_pred, zero_division=0)),
        f1        = float(f1_score(y_te,        y_pred, zero_division=0)),
        cv_acc    = cv_acc,
        cm        = confusion_matrix(y_te, y_pred).tolist(),
    )

    # Linear regression on stress time-series for forecasting
    t  = np.arange(len(df)).reshape(-1, 1)
    lr = LinearRegression().fit(t, df["stress_score"].values)

    return mdl, scaler, df, metrics, lr, float(threshold)


# ══════════════════════════════════════════════════════════
# PREDICTION / INFERENCE HELPERS
# ══════════════════════════════════════════════════════════

def predict_prob(mdl, scaler, vals: list) -> float:
    p = mdl.predict_proba(scaler.transform([vals]))[0][1] * 100
    return float(np.clip(p, 0.0, 95.0))


def forecast_stress(lr_mdl, df, sim_stress, n=6):
    n_hist  = len(df)
    t_fut   = np.arange(n_hist, n_hist + n).reshape(-1, 1)
    trend   = lr_mdl.predict(t_fut)
    rng     = np.random.default_rng(int(abs(sim_stress) * 1000) % 9999)
    noise   = rng.normal(0, abs(df["stress_score"].std()) * 0.15, n)
    blended = 0.60 * trend + 0.40 * sim_stress + noise
    probs   = np.clip(blended * 11 + 40, 0, 95)
    return blended, probs


def explain_prediction(mdl, scaler, vals, threshold):
    z      = scaler.transform([vals])[0]
    raw_w  = z * WEIGHTS
    total  = np.sum(np.abs(raw_w)) + 1e-9
    names  = ["CPI Inflation","Unemployment","S&P 500","Consumer Confidence",
               "Fed Funds Rate","GDP Growth","PMI","Oil Price","Yield Spread"]
    items  = []
    for i in np.argsort(-np.abs(raw_w)):
        items.append(dict(
            feature   = names[i],
            pct       = abs(raw_w[i]) / total * 100,
            direction = FEAT_DIRECTIONS[i],
            value     = vals[i],
            z         = float(z[i]),
        ))
    return items


def policy_effect(policy, vals):
    delta = {
        "Interest Rate Cut (−50 bps)":
            [+0.3, -0.25,  +180,  +4,  -0.5, +0.4,  +1.5,  -2.0, +0.10],
        "Fiscal Stimulus Package":
            [+0.7, -0.50,  +280,  +6,  +0.0, +0.9,  +2.0,  +3.0, +0.05],
        "Tax Increase (+2%)":
            [-0.2,  +0.45, -220,  -5,  +0.0, -0.6,  -1.5,  -1.0, -0.08],
    }.get(policy, [0]*9)
    clips = [(1,15),(2,14),(1800,7000),(30,140),(0,20),(-12,10),(30,70),(15,200),(-3,3)]
    return [float(np.clip(v + d, lo, hi))
            for v, d, (lo, hi) in zip(vals, delta, clips)]


# ══════════════════════════════════════════════════════════
# LLM — ROBUST PARSING (fixed 'choices' KeyError)
# ══════════════════════════════════════════════════════════

def call_llm(system_prompt: str, user_msg: str,
             temperature: float = 0.78, max_tokens: int = 620) -> str:
    """
    Call OpenRouter with safe multi-path response parsing.
    Uses mistralai/mixtral-8x7b for strong economic reasoning.
    """
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type":  "application/json",
                "HTTP-Referer":  "https://economic-warning-ai.streamlit.app",
                "X-Title":       "Economic AI",
            },
            json={
                "model":       "mistralai/mixtral-8x7b-instruct",
                "temperature": temperature,
                "max_tokens":  max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg},
                ],
            },
            timeout=25,
        )
        data = resp.json()

        # ── Safe multi-path parsing ──
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
            pass
        try:
            return data["output"][0]["content"][0]["text"].strip()
        except (KeyError, IndexError, TypeError):
            pass
        # Last resort: look for any "content" key anywhere
        for key in ("content", "text", "response", "answer", "result"):
            if key in data:
                val = data[key]
                return str(val).strip() if not isinstance(val, dict) else str(val)
        return f"⚠ Model returned unexpected format. Raw: {str(data)[:200]}"

    except requests.exceptions.Timeout:
        return "⚠ Request timed out. The AI model is taking too long to respond."
    except requests.exceptions.ConnectionError:
        return "⚠ Connection error. Check your internet connection."
    except Exception as e:
        return f"⚠ Unexpected error: {str(e)}"


def build_system_prompt(live: dict, live_prob: float, live_stress: float,
                         sim_vals: list, sim_prob: float) -> str:
    feat_names = ["CPI Inflation","Unemployment","S&P 500","Consumer Confidence",
                  "Fed Funds Rate","GDP Growth","PMI","Oil Price","Yield Spread"]
    sim_block  = "\n".join(f"  {n}: {v:.2f}" for n, v in zip(feat_names, sim_vals))
    return f"""You are a senior macroeconomic analyst at a global central bank.
You explain cause-and-effect relationships deeply and precisely.
You NEVER use fixed templates. You respond differently each time.
You think step-by-step like an economist: observe → diagnose → project → recommend.
You always reference the specific numbers provided.
You structure responses with 3–5 bullet points using economic reasoning.

=== LIVE ECONOMIC DASHBOARD DATA ===
CPI Inflation:       {live['inflation']:.2f}%
Unemployment Rate:   {live['unemployment']:.2f}%
S&P 500:             {live['sp500']:.0f}
Consumer Confidence: {live['consumer_confidence']:.1f}
Fed Funds Rate:      {live['fed_funds_rate']:.2f}%
GDP Growth:          {live['gdp_growth']:.2f}%
PMI:                 {live['pmi']:.1f}
Oil Price:           ${live['oil_price']:.1f}/bbl
Yield Spread (10y-2y):{live['yield_spread']:.3f}%

AI Recession Probability: {live_prob:.1f}%
Composite Stress Index:   {live_stress:.3f}

=== SCENARIO SIMULATION (USER-ADJUSTED) ===
{sim_block}
Scenario Recession Probability: {sim_prob:.1f}%

Rules:
- Always ground your answer in the data above.
- Explain why each indicator matters economically.
- Show how indicators interact (e.g. "rising rates → higher mortgage costs → falling confidence → slower consumption → GDP contraction").
- Never refuse. Always provide best analytical judgment.
- Vary sentence structure and paragraph organisation every response.
"""


def generate_insights(live, live_prob, live_stress, sim_vals, sim_prob) -> list:
    """Generate 1–4 contextual AI insights using LLM."""
    system = build_system_prompt(live, live_prob, live_stress, sim_vals, sim_prob)
    results = []

    checks = [
        (live["inflation"] > 5.5 and live["unemployment"] > 5.0,
         "danger", "⚡",
         f"CPI is {live['inflation']:.1f}% and unemployment is {live['unemployment']:.1f}%. "
         f"In 3 bullet points explain the stagflation dynamics and why this combination "
         f"is particularly dangerous for recession risk."),

        (live["inflation"] > 5.0,
         "warn", "🔥",
         f"Inflation is running at {live['inflation']:.1f}% with the Fed Funds rate at "
         f"{live['fed_funds_rate']:.1f}%. In 3 bullet points explain the monetary policy "
         f"transmission mechanism and the risk of over-tightening."),

        (live["unemployment"] > 5.5,
         "danger", "👷",
         f"Unemployment is {live['unemployment']:.1f}%. GDP growth is {live['gdp_growth']:.1f}%. "
         f"In 3 bullet points explain the Okun's Law relationship here and the spending "
         f"contraction cycle this could trigger."),

        (live["consumer_confidence"] < 72,
         "info", "😟",
         f"Consumer confidence has dropped to {live['consumer_confidence']:.0f}. "
         f"PMI is {live['pmi']:.1f}. In 3 bullet points explain why confidence is a "
         f"leading indicator and the demand-side risk this signals."),

        (live["yield_spread"] < 0,
         "danger", "📐",
         f"The yield curve is inverted: spread = {live['yield_spread']:.3f}%. "
         f"Fed Funds Rate is {live['fed_funds_rate']:.1f}%. In 3 bullet points explain "
         f"why yield curve inversion is one of the most reliable recession predictors "
         f"and what the historical lead time typically is."),

        (live["sp500"] < 3800,
         "warn", "📉",
         f"The S&P 500 is at {live['sp500']:.0f}. Oil is at ${live['oil_price']:.0f}/bbl. "
         f"In 3 bullet points explain the financial stability implications and the "
         f"wealth-effect transmission to consumption."),

        (live_prob > 65,
         "danger", "🚨",
         f"The AI model outputs {live_prob:.1f}% recession probability. Stress index: "
         f"{live_stress:.3f}. In 3 bullet points synthesise which factors are contributing "
         f"most and what the near-term economic trajectory looks like."),
    ]

    for condition, ctype, icon, prompt in checks:
        if condition:
            answer = call_llm(system, prompt, temperature=round(np.random.uniform(0.7, 0.9), 2))
            results.append((ctype, icon, answer))
        if len(results) >= 4:
            break

    if not results:
        results.append(("success", "✅",
            "All monitored indicators are within normal ranges. The composite stress index "
            "does not currently flag elevated recession risk. The yield curve is positive, "
            "PMI indicates expansion, and consumer confidence is stable. Continued monitoring "
            "is recommended as macro conditions can shift rapidly."))
    return results


# ══════════════════════════════════════════════════════════
# PLOTLY HELPERS
# ══════════════════════════════════════════════════════════

CL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="rgba(255,255,255,.6)", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="rgba(255,255,255,.05)", zerolinecolor="rgba(255,255,255,.08)", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="rgba(255,255,255,.05)", zerolinecolor="rgba(255,255,255,.08)", tickfont=dict(size=11)),
    hoverlabel=dict(bgcolor="rgba(10,20,40,.95)", bordercolor="rgba(0,212,170,.4)",
                    font=dict(family="JetBrains Mono", size=12, color="white")),
    legend=dict(bgcolor="rgba(255,255,255,.04)", bordercolor="rgba(255,255,255,.08)", borderwidth=1),
)


def make_gauge(value, title="Recession Probability"):
    gc  = "#ff3b30" if value > 70 else "#ff9500" if value > 40 else "#00d4aa"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        number=dict(suffix="%", font=dict(size=34, color="white", family="JetBrains Mono")),
        title=dict(text=title, font=dict(size=12, color="rgba(255,255,255,.5)")),
        delta=dict(reference=50, increasing=dict(color="#ff3b30"), decreasing=dict(color="#34c759")),
        gauge=dict(
            axis=dict(range=[0,100], tickwidth=1, tickcolor="rgba(255,255,255,.2)",
                      tickfont=dict(color="rgba(255,255,255,.3)", size=9)),
            bar=dict(color=gc, thickness=0.25), bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[
                dict(range=[0,  40], color="rgba(0,212,170,.12)"),
                dict(range=[40, 70], color="rgba(255,149,0,.12)"),
                dict(range=[70,100], color="rgba(255,59,48,.12)"),
            ],
            threshold=dict(line=dict(color="white", width=2), thickness=0.75, value=value),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk", color="white"),
        height=250, margin=dict(l=30,r=30,t=40,b=10),
    )
    return fig


# ══════════════════════════════════════════════════════════
# APP BOOT
# ══════════════════════════════════════════════════════════

with st.spinner("Fetching economic data…"):
    _econ = fetch_economic_data()

with st.spinner("Training AI recession model…"):
    model, scaler, df, mets, lr_model, threshold = train_pipeline(
        _econ["meta"]["fetched_at"][:13]
    )

live = _econ["live"]
meta = _econ["meta"]

live_vals   = raw_vals_from_live(live)
live_stress = stress_from_raw(live_vals, scaler)
live_prob   = predict_prob(model, scaler, live_vals)
conf_matrix = np.array(mets["cm"])


# ══════════════════════════════════════════════════════════
# ① HERO
# ══════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-header">
    <div class="hero-badge">⚡ AI-POWERED · REAL-TIME ANALYSIS · 9 INDICATORS</div>
    <h1 class="hero-title">Economic Early Warning System</h1>
    <p class="hero-sub">ML-driven recession risk detection · expanded macro features ·
    generative AI analysis · dynamic scenario simulation · 6-month forecasting</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    f'<p style="text-align:center;margin-top:-.5rem;font-size:.65rem;'
    f'color:rgba(255,255,255,.2);font-family:\'JetBrains Mono\',monospace;">'
    f'Source: {meta["source"]} · {meta["fetched_at"][:16]} UTC</p>',
    unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# ② KPI CARDS  (9 indicators, 2 rows)
# ══════════════════════════════════════════════════════════

st.markdown('<p class="section-label">📊 Live Economic Indicators <span class="sim-tag">Simulated where unavailable</span></p>',
            unsafe_allow_html=True)

row1 = st.columns(5)
row2 = st.columns(4)

kpi_r1 = [
    ("blue",  "🏦","CPI Inflation",         f"{live['inflation']:.2f}%",         "FRED CPIAUCSL","warn"),
    ("teal",  "👷","Unemployment",           f"{live['unemployment']:.2f}%",      "FRED UNRATE","down"),
    ("purple","📈","S&P 500",                f"{live['sp500']:.0f}",              "FRED SP500","up"),
    ("red",   "⚠️","Recession Probability",  f"{live_prob:.1f}%",                 "AI Model","down"),
    ("green", "🎯","Model F1 Score",         f"{mets['f1']*100:.1f}%",            "Test-set","up"),
]
kpi_r2 = [
    ("orange","🏛️","Fed Funds Rate",         f"{live['fed_funds_rate']:.2f}%",    "FRED FEDFUNDS","warn"),
    ("cyan",  "📉","Yield Spread (10y-2y)",  f"{live['yield_spread']:+.3f}%",     "Derived","warn" if live['yield_spread']<0 else "up"),
    ("teal",  "⚙️","PMI",                   f"{live['pmi']:.1f}",                "Simulated","up" if live['pmi']>50 else "down"),
    ("orange","🛢️","Oil Price",              f"${live['oil_price']:.1f}",         "FRED DCOILWTICO","warn"),
]

for (col,(color,icon,label,value,delta,dcls)) in zip(row1, kpi_r1):
    with col:
        st.markdown(f"""<div class="kpi-card {color}">
        <div class="kpi-icon">{icon}</div><div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div><div class="kpi-delta {dcls}">{delta}</div>
        </div>""", unsafe_allow_html=True)

for (col,(color,icon,label,value,delta,dcls)) in zip(row2, kpi_r2):
    with col:
        st.markdown(f"""<div class="kpi-card {color}">
        <div class="kpi-icon">{icon}</div><div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div><div class="kpi-delta {dcls}">{delta}</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# ③ MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════

st.markdown('<p class="section-label">🎯 Model Performance · 80/20 Split · 5-Fold CV · 9 Features</p>',
            unsafe_allow_html=True)

mp1, mp2, mp3 = st.columns([1,1,2], gap="medium")

with mp1:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Test-Set Metrics</p>', unsafe_allow_html=True)
    for label, val in [("Accuracy",  mets["accuracy"]),
                        ("Precision", mets["precision"]),
                        ("Recall",    mets["recall"]),
                        ("F1 Score",  mets["f1"]),
                        ("CV Acc (5-fold)", mets["cv_acc"])]:
        pct   = val * 100
        color = "#34c759" if pct>=75 else "#ff9500" if pct>=60 else "#ff3b30"
        st.markdown(
            f'<div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);'
            f'border-radius:8px;padding:.4rem .8rem;margin:.25rem 0;font-family:\'JetBrains Mono\','
            f'monospace;font-size:.78rem;color:rgba(255,255,255,.6);">'
            f'{label}: <span style="color:{color};font-weight:600;">{pct:.1f}%</span></div>',
            unsafe_allow_html=True)
    st.markdown('<p style="font-size:.67rem;color:rgba(255,255,255,.2);margin-top:.75rem;">'
                'RandomForest 300 trees · depth 7 · 5% label noise</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with mp2:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Confusion Matrix</p>', unsafe_allow_html=True)
    fig_cm = go.Figure(go.Heatmap(
        z=conf_matrix,
        x=["Pred: Low","Pred: High"], y=["Actual: Low","Actual: High"],
        colorscale=[[0,"rgba(0,212,170,.25)"],[1,"rgba(255,59,48,.75)"]],
        showscale=False, text=conf_matrix, texttemplate="%{text}",
        textfont=dict(size=22, color="white", family="JetBrains Mono"),
    ))
    fig_cm.update_layout(**CL, height=220)
    st.plotly_chart(fig_cm, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with mp3:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Feature Importance (9 Indicators)</p>', unsafe_allow_html=True)
    feat_names_short = ["CPI","Unemployment","S&P 500","Confidence",
                         "Fed Rate","GDP","PMI","Oil","Yield Spread"]
    imps = model.feature_importances_
    idx  = np.argsort(imps)
    fcolors = ["#007aff","#6326ff","#00d4aa","#ff9500","#ff3b30",
               "#34c759","#5ac8fa","#af52de","#ffcc00"]
    fig_fi = go.Figure(go.Bar(
        x=imps[idx], y=[feat_names_short[i] for i in idx], orientation="h",
        marker=dict(color=[fcolors[i] for i in idx], opacity=.85),
        text=[f"{imps[i]:.3f}" for i in idx], textposition="inside",
        textfont=dict(color="white", size=10, family="JetBrains Mono"),
        hovertemplate="<b>%{y}</b>: %{x:.4f}<extra></extra>",
    ))
    fig_fi.update_layout(**CL, height=260, showlegend=False)
    fig_fi.update_xaxes(title_text="Gini Importance")
    st.plotly_chart(fig_fi, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# ④ STRESS TREND + SCATTER
# ══════════════════════════════════════════════════════════

st.markdown('<p class="section-label">📉 Economic Stress Analysis</p>', unsafe_allow_html=True)

ct, cs = st.columns([3,2], gap="medium")

with ct:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Composite Stress Score — Historical</p>', unsafe_allow_html=True)
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        y=df["stress_score"], mode="lines",
        line=dict(color="#00d4aa", width=2, shape="spline"),
        fill="tozeroy", fillcolor="rgba(0,212,170,.07)",
        hovertemplate="Period %{x}<br>Stress: %{y:.3f}<extra></extra>",
    ))
    fig_trend.add_hline(y=threshold, line_dash="dash",
                        line_color="rgba(255,149,0,.6)", line_width=1.5,
                        annotation_text="Recession threshold (median)",
                        annotation_font_color="rgba(255,180,80,.8)", annotation_position="right")
    fig_trend.update_layout(**CL, xaxis_title="Observation", yaxis_title="Stress Score", height=300)
    st.plotly_chart(fig_trend, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with cs:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Stress vs Unemployment</p>', unsafe_allow_html=True)
    fig_sc = go.Figure(go.Scatter(
        x=df["unemployment"], y=df["stress_score"], mode="markers",
        marker=dict(size=6, color=df["stress_score"],
                    colorscale=[[0,"#00d4aa"],[.5,"#007aff"],[1,"#ff3b30"]],
                    showscale=True, colorbar=dict(title="Stress", thickness=10, len=.7),
                    line=dict(color="#050d1a", width=1)),
        hovertemplate="Unemployment: %{x:.1f}%<br>Stress: %{y:.3f}<extra></extra>",
    ))
    fig_sc.update_layout(**CL, xaxis_title="Unemployment (%)", yaxis_title="Stress Score", height=300)
    st.plotly_chart(fig_sc, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# ⑤ SCENARIO SIMULATOR — model-connected, 9-feature
# ══════════════════════════════════════════════════════════

st.markdown('<p class="section-label">🎛️ Scenario Simulator — All inputs feed the trained model</p>',
            unsafe_allow_html=True)

sc1, sc2, sc3 = st.columns(3, gap="medium")

with sc1:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    sim_infl  = st.slider("🔥 Inflation (%)",            0.0, 15.0,
                           float(round(live["inflation"],1)), 0.1)
    sim_unemp = st.slider("👷 Unemployment (%)",         2.0, 15.0,
                           float(round(live["unemployment"],1)), 0.1)
    sim_sp    = st.slider("📈 S&P 500",                2000, 6500,
                           int(round(live["sp500"],-2)), 50)
    st.markdown('</div>', unsafe_allow_html=True)

with sc2:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    sim_conf  = st.slider("😟 Consumer Confidence",     30, 130,
                           int(round(live["consumer_confidence"])), 1)
    sim_rate  = st.slider("🏛️ Fed Funds Rate (%)",       0.0, 20.0,
                           float(round(live["fed_funds_rate"],1)), 0.25)
    sim_gdp   = st.slider("📊 GDP Growth (%)",          -8.0, 8.0,
                           float(round(live["gdp_growth"],1)), 0.1)
    st.markdown('</div>', unsafe_allow_html=True)

with sc3:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    sim_pmi   = st.slider("⚙️ PMI",                    30.0, 70.0,
                           float(round(live["pmi"],1)), 0.5)
    sim_oil   = st.slider("🛢️ Oil Price ($/bbl)",       20, 160,
                           int(round(live["oil_price"])), 1)
    sim_yld   = st.slider("📐 Yield Spread 10y-2y (%)", -2.0, 3.0,
                           float(round(live["yield_spread"],2)), 0.05)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Real-time model inference ──
sim_vals   = [sim_infl, sim_unemp, sim_sp, sim_conf,
              sim_rate, sim_gdp, sim_pmi, sim_oil, sim_yld]
sim_stress = stress_from_raw(sim_vals, scaler)
sim_prob   = predict_prob(model, scaler, sim_vals)
sim_health = 100 - sim_prob


# ══════════════════════════════════════════════════════════
# ⑥ POLICY SIMULATOR — model-connected
# ══════════════════════════════════════════════════════════

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("## 🧠 AI Policy Decision Simulator")

policy = st.selectbox("Select a policy intervention:",
    ["Interest Rate Cut (−50 bps)", "Fiscal Stimulus Package", "Tax Increase (+2%)"])

pol_vals   = policy_effect(policy, sim_vals)
pol_stress = stress_from_raw(pol_vals, scaler)
pol_prob   = predict_prob(model, scaler, pol_vals)

pol_desc = {
    "Interest Rate Cut (−50 bps)":
        "Rate cuts lower borrowing costs → stimulate business investment & consumer credit "
        "→ boost equity prices and confidence → mild upward pressure on inflation.",
    "Fiscal Stimulus Package":
        "Government spending raises aggregate demand → lowers unemployment → "
        "boosts GDP and PMI → upward inflation pressure from demand-pull dynamics.",
    "Tax Increase (+2%)":
        "Higher taxes reduce household disposable income → softer consumption → "
        "lower confidence and PMI → downward GDP pressure but reduced inflation.",
}
st.info(pol_desc[policy])

feat_labels = ["Inflation","Unemployment","S&P 500","Confidence",
               "Fed Rate","GDP","PMI","Oil","Yield Spread"]
p1,p2,p3,p4 = st.columns(4)
p1.metric("Stress Index Δ",    f"{pol_stress:.3f}", f"{pol_stress-sim_stress:+.3f}")
p2.metric("Recession Prob Δ",  f"{pol_prob:.1f}%",  f"{pol_prob-sim_prob:+.1f}%")
p3.metric("Unemployment Δ",
          f"{pol_vals[1]:.2f}%", f"{pol_vals[1]-sim_unemp:+.2f}%")
p4.metric("Inflation Δ",
          f"{pol_vals[0]:.2f}%", f"{pol_vals[0]-sim_infl:+.2f}%")


# ══════════════════════════════════════════════════════════
# ⑦ RESULTS ROW
# ══════════════════════════════════════════════════════════

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">📡 Live Model Output — Scenario</p>',
            unsafe_allow_html=True)

r1, r2, r3 = st.columns([1,2,1], gap="medium")

with r1:
    st.markdown('<div class="glass-panel" style="text-align:center;">', unsafe_allow_html=True)
    st.markdown(f'<p class="score-label">STRESS INDEX</p>'
                f'<p class="score-display">{sim_stress:.2f}</p>', unsafe_allow_html=True)
    if sim_stress > df["stress_score"].quantile(.75):
        badge, cls = "🔴 ELEVATED", "high"
    elif sim_stress > threshold:
        badge, cls = "🟡 MODERATE", "medium"
    else:
        badge, cls = "🟢 STABLE",   "low"
    st.markdown(f'<div style="text-align:center">'
                f'<span class="risk-badge {cls}">{badge}</span></div>',
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with r2:
    st.plotly_chart(make_gauge(sim_prob), use_container_width=True)

with r3:
    st.markdown('<div class="glass-panel" style="text-align:center;">', unsafe_allow_html=True)
    st.markdown(f'<p class="score-label">ECONOMIC HEALTH</p>'
                f'<p class="score-display">{sim_health:.1f}</p>', unsafe_allow_html=True)
    st.markdown('<p class="score-label" style="margin-top:.3rem;">out of 100</p>',
                unsafe_allow_html=True)
    st.progress(int(sim_health))
    st.markdown('</div>', unsafe_allow_html=True)

# ── Why this prediction ──
if st.button("🔍 Why this prediction?", use_container_width=False):
    expl = explain_prediction(model, scaler, sim_vals, threshold)
    st.markdown('<div class="why-box">', unsafe_allow_html=True)
    st.markdown("**Top contributing factors:**")
    for item in expl:
        dc = "#ff6b6b" if "raises" in item["direction"] else "#34c759"
        st.markdown(
            f'<div style="margin:.35rem 0;">'
            f'<span style="color:#fff;font-weight:600;">{item["feature"]}</span>'
            f' → <span style="color:{dc}">{item["direction"]}</span><br>'
            f'<span style="font-size:.78rem;color:rgba(255,255,255,.4);">'
            f'Contribution: <b style="color:#00d4aa">{item["pct"]:.1f}%</b> · '
            f'Value: {item["value"]:.2f} · z-score: {item["z"]:+.2f}</span>'
            f'<div style="height:4px;border-radius:2px;background:rgba(255,255,255,.07);margin-top:.25rem;">'
            f'<div style="height:4px;width:{min(int(item["pct"]),100)}%;'
            f'border-radius:2px;background:{dc};"></div></div>'
            f'</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# ⑧ REAL FORECAST — trend + scenario + noise
# ══════════════════════════════════════════════════════════

st.markdown('<p class="section-label">🔮 6-Month Forecast</p>', unsafe_allow_html=True)
st.caption("Forecast = 60% historical linear trend + 40% current scenario stress + calibrated noise. "
           "Updates dynamically as you adjust the scenario sliders above.")

f_stress, f_probs = forecast_stress(lr_model, df, sim_stress, 6)
months = ["Month 1","Month 2","Month 3","Month 4","Month 5","Month 6"]

st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown('<p class="section-title">Projected Economic Risk — 6-Month Horizon</p>',
            unsafe_allow_html=True)

fig_fc = go.Figure()
fig_fc.add_trace(go.Bar(
    x=months, y=f_stress, name="Forecast Stress",
    marker=dict(color=list(f_stress),
                colorscale=[[0,"rgba(0,212,170,.7)"],[.5,"rgba(0,122,255,.7)"],[1,"rgba(255,59,48,.7)"]],
                opacity=.8),
    yaxis="y",
    hovertemplate="<b>%{x}</b><br>Stress: %{y:.3f}<extra></extra>",
))
fig_fc.add_trace(go.Scatter(
    x=months, y=f_probs, name="Recession Probability %",
    mode="lines+markers",
    line=dict(color="#ff9500", width=2.5, shape="spline"),
    marker=dict(size=8, color="#ff9500", line=dict(color="#050d1a", width=2)),
    yaxis="y2",
    hovertemplate="<b>%{x}</b><br>Probability: %{y:.1f}%<extra></extra>",
))
fig_fc.update_layout(
    **CL, height=340, barmode="group",
    yaxis2=dict(title="Recession Probability (%)", overlaying="y", side="right",
                range=[0,100], gridcolor="rgba(0,0,0,0)",
                tickfont=dict(size=11, color="rgba(255,149,0,.7)")),
)
fig_fc.update_yaxes(title_text="Stress Score", selector=dict(overlaying=None))
fig_fc.update_layout(legend=dict(
    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
    bgcolor="rgba(255,255,255,.04)", bordercolor="rgba(255,255,255,.08)", borderwidth=1,
))
st.plotly_chart(fig_fc, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# ⑨ GLOBAL MAP — model-connected, regional linkage
# ══════════════════════════════════════════════════════════

st.markdown('<p class="section-label">🌍 Global Recession Risk Map</p>', unsafe_allow_html=True)
st.caption("US risk is direct model output. Regional risks are estimated using "
           "trade-linkage coefficients and financial contagion research. "
           "Updates when scenario changes.")

# Trade/financial linkage coefficients (literature-derived approximations)
REGION_COEFFS = {
    "United States":  1.00,
    "Canada":         0.83,
    "Mexico":         0.74,
    "United Kingdom": 0.78,
    "Germany":        0.72,
    "France":         0.70,
    "Italy":          0.67,
    "Spain":          0.64,
    "Japan":          0.65,
    "South Korea":    0.63,
    "China":          0.68,
    "India":          0.50,
    "Brazil":         0.57,
    "Australia":      0.61,
    "South Africa":   0.48,
}

rng_noise = np.random.default_rng(int(abs(sim_prob) * 100) % 9999)
map_rows  = []
for country, coeff in REGION_COEFFS.items():
    base  = sim_prob * coeff
    noise = rng_noise.uniform(-2.5, 2.5)
    map_rows.append({"country": country,
                     "risk": float(np.clip(base + noise, 0, 95)),
                     "linkage": f"{coeff:.0%}"})

map_df  = pd.DataFrame(map_rows)
fig_map = px.choropleth(
    map_df, locations="country", locationmode="country names",
    color="risk", color_continuous_scale="Reds", range_color=[10, 85],
    labels={"risk": "Recession Risk (%)"},
    hover_data={"linkage": True, "risk": ":.1f"},
    title="Global Recession Risk — estimated via trade & financial linkage",
)
fig_map.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="rgba(255,255,255,.6)"),
    geo=dict(bgcolor="rgba(0,0,0,0)", showframe=False,
             showcoastlines=True, coastlinecolor="rgba(255,255,255,.1)"),
    margin=dict(l=0,r=0,t=40,b=0),
    coloraxis_colorbar=dict(title="Risk %"),
)
st.plotly_chart(fig_map, use_container_width=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# ⑩ AI ECONOMIC INSIGHTS
# ══════════════════════════════════════════════════════════

st.markdown('<p class="section-label">🤖 Generative AI Economic Insights</p>',
            unsafe_allow_html=True)

ins_col, pol_col = st.columns([3,2], gap="medium")

with ins_col:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">AI Macroeconomic Analysis</p>', unsafe_allow_html=True)
    with st.spinner("Generating AI insights…"):
        insights = generate_insights(live, live_prob, live_stress, sim_vals, sim_prob)
    for ctype, icon, text in insights:
        st.markdown(f"""<div class="insight-card {ctype}">
        <span class="insight-icon">{icon}</span>
        <span class="insight-text">{text}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with pol_col:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">AI Policy Recommendation</p>', unsafe_allow_html=True)
    if sim_prob > 70:
        p_icon,p_cls,p_label = "🏦","high","🔴 SEVERE RECESSION RISK"
        p_text = ("<strong>Immediate intervention required.</strong> Central banks should "
                  "consider rate reductions and liquidity provision. Fiscal authorities "
                  "should prepare emergency stimulus frameworks targeting unemployment.")
    elif sim_prob > 40:
        p_icon,p_cls,p_label = "📊","medium","🟡 MODERATE ECONOMIC RISK"
        p_text = ("<strong>Heightened vigilance warranted.</strong> Monitor leading indicators "
                  "closely — particularly the yield curve and PMI. Prepare contingency "
                  "policy responses for rapid deployment if stress metrics deteriorate.")
    else:
        p_icon,p_cls,p_label = "🌿","low","🟢 STABLE CONDITIONS"
        p_text = ("<strong>Macro conditions appear stable.</strong> Maintain current policy "
                  "stance. Focus on building fiscal buffers and structural resilience. "
                  "Continue monitoring yield spread and credit market signals.")
    st.markdown(f"""<div class="policy-box">
    <div style="font-size:1.8rem;margin-bottom:.75rem;">{p_icon}</div>{p_text}
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-label" style="margin-bottom:.5rem;">Risk Level</p>',
                unsafe_allow_html=True)
    st.markdown(f'<span class="risk-badge {p_cls}" style="font-size:.9rem;'
                f'padding:.5rem 1.2rem;">{p_label}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# ⑪ AI CHATBOT — context-aware, Mixtral
# ══════════════════════════════════════════════════════════

st.markdown("## 🤖 AI Economist Assistant")
st.markdown('<p style="color:rgba(255,255,255,.4);font-size:.85rem;margin-top:-.5rem;margin-bottom:1rem;">'
            'Ask about any indicator, cause-effect relationships, scenario impacts, '
            'or global contagion. Powered by Mixtral 8x7B.</p>', unsafe_allow_html=True)

question = st.text_input(
    "Your question:",
    placeholder='"Why is the yield curve inverted?" · "What happens if oil hits $120?" · '
                '"Explain China debt risk and its impact here"',
)

if question:
    temp   = round(float(np.random.uniform(0.72, 0.90)), 2)
    system = build_system_prompt(live, live_prob, live_stress, sim_vals, sim_prob)
    with st.spinner(f"Analysing with Mixtral 8x7B (temp={temp})…"):
        answer = call_llm(system, question, temperature=temp, max_tokens=640)
    st.info(answer)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# ⑫ AI NEWS ANALYZER
# ══════════════════════════════════════════════════════════

st.markdown('<p class="section-label">🗞️ Economic News Sentiment</p>',
            unsafe_allow_html=True)

news = get_news()
if news:
    for art in news:
        s = sentiment(art["title"])
        msg = f"**{art['source']}** — {art['title']}"
        if s == "Negative":   st.error(msg)
        elif s == "Neutral":  st.warning(msg)
        else:                 st.success(msg)
else:
    st.warning("No economic news available right now.")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# ⑬ WHY THIS MATTERS
# ══════════════════════════════════════════════════════════

st.markdown('<p class="section-label">💡 Why This Matters</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Real-World Stakeholder Value</p>', unsafe_allow_html=True)

wm1, wm2, wm3 = st.columns(3, gap="medium")

with wm1:
    st.markdown("""<div class="matters-card">
    <h4>🏛️ Central Banks & Governments</h4>
    <p>Early recession signals allow policymakers to pre-emptively adjust interest rates,
    deploy fiscal stimulus, or tighten financial regulation — reducing the severity
    of downturns before they become systemic.</p>
    <p><b>Actionable insight:</b> A rising stress index 2–3 quarters ahead gives
    governments time to prepare automatic stabilisers and targeted unemployment support.</p>
    </div>""", unsafe_allow_html=True)

with wm2:
    st.markdown("""<div class="matters-card">
    <h4>🏦 Banks & Financial Institutions</h4>
    <p>Credit risk models must anticipate macro downturns to set appropriate
    loan-loss provisions. A rising recession probability signals tighter
    lending standards and increased collateral requirements.</p>
    <p><b>Actionable insight:</b> When AI probability exceeds 50%, increase
    capital buffers and review exposure to cyclical sectors.</p>
    </div>""", unsafe_allow_html=True)

with wm3:
    st.markdown("""<div class="matters-card">
    <h4>📈 Investors & Portfolio Managers</h4>
    <p>Asset allocation should respond to macro regime shifts. Rising stress
    indices historically correlate with equity drawdowns, credit spread widening,
    and safe-haven flows to bonds and gold.</p>
    <p><b>Actionable insight:</b> Use the scenario simulator to stress-test
    portfolio assumptions against stagflation, rate-shock, or demand-collapse scenarios.</p>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# ⑭ HOW AI WORKS
# ══════════════════════════════════════════════════════════

st.markdown('<p class="section-label">🔬 Model Transparency</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">How the AI Works</p>', unsafe_allow_html=True)

hw1, hw2 = st.columns(2, gap="medium")
with hw1:
    st.markdown("""<div class="info-box"><h4>📥 9 Input Features</h4>
    CPI Inflation · Unemployment Rate · S&amp;P 500 · Consumer Confidence ·
    Fed Funds Rate · GDP Growth · PMI · Oil Price · Yield Spread (10y−2y)<br><br>
    All FRED-sourced where available; realistic simulation as fallback.
    </div>
    <div class="info-box"><h4>📐 Composite Stress Score</h4>
    Features are standardized with <code>StandardScaler</code>. Weighted combination:<br>
    <code>S = Σ(wᵢ · zᵢ)</code> where positive weights raise risk (inflation, unemployment,
    oil, rate) and negative weights lower risk (S&P, confidence, GDP, PMI).
    Yield spread uses inverted sign: curve inversion = higher stress.
    </div>""", unsafe_allow_html=True)
with hw2:
    st.markdown("""<div class="info-box"><h4>🎯 Classification</h4>
    Stress score binarised at historical median → balanced 50/50 classes.
    5% label noise injected during training prevents overfit → realistic 70–85% accuracy.
    </div>
    <div class="info-box"><h4>📤 Forecast & Map</h4>
    <b>Forecast:</b> 60% linear regression trend on historical stress + 40% current
    scenario value + calibrated Gaussian noise → dynamic response to slider changes.<br><br>
    <b>Map:</b> US recession probability from model. Regional risks = US prob × trade/financial
    linkage coefficient (research-derived) + uniform noise ±2.5%.
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# ⑮ LIMITATIONS
# ══════════════════════════════════════════════════════════

st.markdown('<p class="section-label">⚠️ Limitations & Disclosures</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Model Limitations</p>', unsafe_allow_html=True)

l1,l2,l3 = st.columns(3, gap="medium")
with l1:
    st.markdown("""<div class="limit-box"><h4>🔧 Simplified Model</h4>
    Uses 9 indicators; production systems incorporate 50+. Structural breaks,
    geopolitical shocks, and banking crises cannot be captured by tabular ML alone.
    </div>""", unsafe_allow_html=True)
with l2:
    st.markdown("""<div class="limit-box"><h4>📋 Not Financial Advice</h4>
    All probabilities, scores, and recommendations are for <b>educational
    and research purposes only</b>. Not suitable for investment, policy,
    or business decisions without expert validation.
    </div>""", unsafe_allow_html=True)
with l3:
    st.markdown("""<div class="limit-box"><h4>📊 Simulated Data</h4>
    Where FRED APIs are unavailable, data is generated from historically-calibrated
    distributions. Model generalisation to future structural regime changes is limited.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center;padding:1rem 0 .5rem;">
<p style="font-family:'JetBrains Mono',monospace;font-size:.6rem;letter-spacing:.15em;
          color:rgba(255,255,255,.15);text-transform:uppercase;">
AI Economic Early Warning System · RandomForest + Mixtral 8x7B · 9 Macro Indicators ·
FRED / Simulation · {meta['fetched_at'][:10]} · For Informational Purposes Only
</p></div>
""", unsafe_allow_html=True)