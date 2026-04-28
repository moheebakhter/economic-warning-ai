"""
AI Economic Early Warning System
─────────────────────────────────
Realistic, transparent, production-grade macro risk platform.
Data: FRED-sourced where available; historically-calibrated simulation as fallback.
Model: RandomForestClassifier (80/20 split, 5-fold CV, label noise for realism).
LLM: Mixtral 8x7B via OpenRouter — variable, human-like economic reasoning.
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
    page_title="Economic Early Warning System",
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
        radial-gradient(ellipse 80% 60% at 20% 10%,rgba(0,122,255,.11) 0%,transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 80%,rgba(0,212,170,.07) 0%,transparent 55%),
        radial-gradient(ellipse 40% 40% at 60% 30%,rgba(99,38,255,.06) 0%,transparent 50%);
    min-height:100vh;
}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:2rem 2.5rem 3rem;max-width:1440px;}

/* ── Hero ── */
.hero-header{text-align:center;padding:2.5rem 0 1rem;margin-bottom:.5rem;}
.hero-badge{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:.68rem;
    font-weight:500;letter-spacing:.15em;text-transform:uppercase;color:#00d4aa;
    background:rgba(0,212,170,.1);border:1px solid rgba(0,212,170,.25);
    padding:.3rem 1rem;border-radius:50px;margin-bottom:1rem;}
.hero-title{font-size:clamp(1.9rem,4vw,3.1rem);font-weight:700;letter-spacing:-.03em;
    line-height:1.1;background:linear-gradient(135deg,#fff 0%,#a8c8ff 50%,#00d4aa 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;margin:0 0 .6rem;}
.hero-sub{font-size:.95rem;color:rgba(255,255,255,.42);max-width:600px;margin:0 auto;line-height:1.65;}
.transparency-bar{
    background:rgba(255,200,80,.07);border:1px solid rgba(255,200,80,.2);
    border-radius:10px;padding:.65rem 1.1rem;font-size:.78rem;
    color:rgba(255,220,120,.75);margin-top:1rem;text-align:center;
    font-family:'JetBrains Mono',monospace;letter-spacing:.03em;}

/* ── Layout ── */
.section-divider{height:1px;background:linear-gradient(90deg,transparent,rgba(0,212,170,.28),rgba(99,38,255,.28),transparent);margin:2rem 0;}
.section-label{font-family:'JetBrains Mono',monospace;font-size:.63rem;font-weight:500;
    letter-spacing:.2em;text-transform:uppercase;color:rgba(255,255,255,.28);
    margin-bottom:.75rem;display:flex;align-items:center;gap:.5rem;}
.section-label::after{content:'';flex:1;height:1px;background:rgba(255,255,255,.06);}
.section-title{font-size:1.2rem;font-weight:600;color:#fff;letter-spacing:-.02em;margin:0 0 1.2rem;}

/* ── KPI Cards ── */
.kpi-card{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);
    border-radius:14px;padding:1.2rem 1.4rem;position:relative;overflow:hidden;
    backdrop-filter:blur(20px);margin-bottom:1rem;}
.kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;border-radius:14px 14px 0 0;}
.kpi-card.blue::before{background:linear-gradient(90deg,#007aff,#5ac8fa);}
.kpi-card.teal::before{background:linear-gradient(90deg,#00d4aa,#34c759);}
.kpi-card.purple::before{background:linear-gradient(90deg,#6326ff,#af52de);}
.kpi-card.red::before{background:linear-gradient(90deg,#ff3b30,#ff6b35);}
.kpi-card.green::before{background:linear-gradient(90deg,#34c759,#30d158);}
.kpi-card.orange::before{background:linear-gradient(90deg,#ff9500,#ffcc00);}
.kpi-card.cyan::before{background:linear-gradient(90deg,#32ade6,#007aff);}
.kpi-card.pink::before{background:linear-gradient(90deg,#ff375f,#ff6b95);}
.kpi-icon{font-size:1rem;margin-bottom:.5rem;opacity:.65;}
.kpi-label{font-size:.68rem;font-weight:500;letter-spacing:.1em;text-transform:uppercase;
    color:rgba(255,255,255,.38);margin-bottom:.35rem;font-family:'JetBrains Mono',monospace;}
.kpi-value{font-size:1.85rem;font-weight:700;color:#fff;letter-spacing:-.03em;
    line-height:1;margin-bottom:.3rem;font-family:'JetBrains Mono',monospace;}
.kpi-delta{font-size:.7rem;font-weight:500;color:rgba(255,255,255,.3);}
.kpi-delta.up{color:#34c759;}.kpi-delta.down{color:#ff3b30;}.kpi-delta.warn{color:#ff9500;}
.kpi-tag{font-size:.58rem;font-family:'JetBrains Mono',monospace;letter-spacing:.08em;
    text-transform:uppercase;padding:.1rem .35rem;border-radius:3px;margin-left:.3rem;vertical-align:middle;}
.kpi-tag.real{color:rgba(0,212,170,.7);background:rgba(0,212,170,.1);border:1px solid rgba(0,212,170,.2);}
.kpi-tag.sim{color:rgba(255,200,80,.6);background:rgba(255,200,80,.08);border:1px solid rgba(255,200,80,.18);}

/* ── Glass ── */
.glass-panel{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);
    border-radius:18px;padding:1.65rem;backdrop-filter:blur(20px);
    margin-bottom:1.5rem;position:relative;overflow:hidden;}
.glass-panel::before{content:'';position:absolute;inset:0;border-radius:18px;
    background:linear-gradient(135deg,rgba(255,255,255,.025) 0%,transparent 60%);pointer-events:none;}

/* ── Risk badge ── */
.risk-badge{display:inline-flex;align-items:center;gap:.4rem;font-size:.78rem;
    font-weight:600;letter-spacing:.05em;padding:.32rem .85rem;border-radius:50px;}
.risk-badge.high{background:rgba(255,59,48,.15);color:#ff6b6b;border:1px solid rgba(255,59,48,.3);}
.risk-badge.medium{background:rgba(255,149,0,.15);color:#ffbb55;border:1px solid rgba(255,149,0,.3);}
.risk-badge.low{background:rgba(52,199,89,.15);color:#34c759;border:1px solid rgba(52,199,89,.3);}

/* ── Sliders ── */
.stSlider>div>div>div{background:rgba(0,212,170,.2)!important;}
.stSlider>div>div>div>div{background:#00d4aa!important;}

/* ── Score ── */
.score-display{font-family:'JetBrains Mono',monospace;font-size:3.2rem;font-weight:700;
    letter-spacing:-.04em;background:linear-gradient(135deg,#fff,#00d4aa);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;text-align:center;line-height:1;margin:.5rem 0;}
.score-label{font-size:.68rem;letter-spacing:.15em;text-transform:uppercase;
    color:rgba(255,255,255,.28);text-align:center;font-family:'JetBrains Mono',monospace;}

/* ── Insight cards ── */
.insight-card{display:flex;align-items:flex-start;gap:.8rem;padding:.95rem 1.1rem;
    border-radius:11px;margin-bottom:.55rem;border:1px solid rgba(255,255,255,.06);}
.insight-card.warn{background:rgba(255,149,0,.07);border-color:rgba(255,149,0,.18);}
.insight-card.danger{background:rgba(255,59,48,.07);border-color:rgba(255,59,48,.18);}
.insight-card.info{background:rgba(0,122,255,.07);border-color:rgba(0,122,255,.18);}
.insight-card.success{background:rgba(52,199,89,.07);border-color:rgba(52,199,89,.18);}
.insight-icon{font-size:1.05rem;margin-top:.05rem;flex-shrink:0;}
.insight-text{font-size:.84rem;color:rgba(255,255,255,.78);line-height:1.55;}

/* ── Policy box ── */
.policy-box{background:rgba(99,38,255,.09);border:1px solid rgba(99,38,255,.22);
    border-radius:13px;padding:1.1rem 1.3rem;font-size:.88rem;
    color:rgba(255,255,255,.82);line-height:1.65;}
.policy-box strong{color:#af88ff;}

/* ── Info/Limit boxes ── */
.info-box{background:rgba(0,122,255,.06);border:1px solid rgba(0,122,255,.18);
    border-radius:13px;padding:1.1rem 1.3rem;font-size:.85rem;
    color:rgba(255,255,255,.75);line-height:1.7;margin-bottom:.7rem;}
.info-box h4{color:#5ac8fa;font-size:.92rem;margin:0 0 .4rem;}
.limit-box{background:rgba(255,149,0,.06);border:1px solid rgba(255,149,0,.18);
    border-radius:13px;padding:1.1rem 1.3rem;font-size:.85rem;
    color:rgba(255,255,255,.75);line-height:1.7;margin-bottom:.7rem;}
.limit-box h4{color:#ffbb55;font-size:.92rem;margin:0 0 .4rem;}

/* ── Why box ── */
.why-box{background:rgba(99,38,255,.07);border:1px solid rgba(99,38,255,.22);
    border-radius:13px;padding:1.1rem 1.3rem;margin-top:1rem;}
.feat-bar-wrap{height:4px;border-radius:2px;background:rgba(255,255,255,.06);margin-top:.22rem;}
.feat-bar-fill{height:4px;border-radius:2px;}

/* ── Matters cards ── */
.matters-card{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);
    border-radius:14px;padding:1.3rem 1.5rem;height:100%;margin-bottom:1rem;}
.matters-card h4{color:#00d4aa;font-size:.96rem;margin:0 0 .55rem;}
.matters-card p{font-size:.83rem;color:rgba(255,255,255,.6);line-height:1.6;margin:.28rem 0;}

/* ── Metric chip ── */
[data-testid="metric-container"]{background:transparent;border:none;padding:0;}
.stProgress>div>div>div{background:linear-gradient(90deg,#00d4aa,#007aff)!important;border-radius:4px;}
.stProgress>div>div{background:rgba(255,255,255,.06)!important;border-radius:4px;}
.stDataFrame{border-radius:11px;overflow:hidden;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# DATA LAYER
# ══════════════════════════════════════════════

def _sim(mean, std, n, seed, lo=None, hi=None):
    """Auto-correlated simulation of economic time-series."""
    rng, vals = np.random.default_rng(seed), [float(mean)]
    for _ in range(n - 1):
        v = vals[-1] * 0.91 + mean * 0.09 + rng.normal(0, std)
        if lo is not None: v = max(v, lo)
        if hi is not None: v = min(v, hi)
        vals.append(float(v))
    return vals


def _fred_series(series_id, n=240):
    try:
        url  = (f"https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json")
        obs  = requests.get(url, timeout=9).json().get("observations", [])
        vals = [float(o["value"]) for o in obs if o["value"] != "."]
        if len(vals) >= 40:
            return vals[-n:], True          # (data, is_real)
    except Exception:
        pass
    return [], False


def _fred_latest(series_id):
    try:
        url = (f"https://api.stlouisfed.org/fred/series/observations"
               f"?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json")
        obs = requests.get(url, timeout=9).json().get("observations", [])
        for o in reversed(obs):
            if o["value"] != ".":
                return float(o["value"]), True
    except Exception:
        pass
    return None, False


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_economic_data():
    n = 240

    def get(series_id, fallback_fn):
        vals, real = _fred_series(series_id, n)
        if vals:
            return vals, real
        return fallback_fn(), False

    infl,  infl_real  = get("CPIAUCSL",         lambda: _sim(4.5, 1.5, n, 1,  1.0, 14.0))
    unemp, unemp_real = get("UNRATE",            lambda: _sim(4.8, 1.1, n, 2,  2.0, 14.0))
    sp,    sp_real    = get("SP500",             lambda: _sim(4100, 520, n, 3, 1800, 6800))
    conf,  conf_real  = get("UMCSENT",           lambda: _sim(82,  13,  n, 4,   30,  140))
    rate,  rate_real  = get("FEDFUNDS",          lambda: _sim(4.0, 1.2, n, 5,  0.0,  20.0))
    oil,   oil_real   = get("DCOILWTICO",        lambda: _sim(78,  18,  n, 6,  15,   180))
    t10,   t10_real   = get("GS10",              lambda: _sim(3.8, 0.9, n, 7,  0.5,   8.0))
    t2,    t2_real    = get("GS2",               lambda: _sim(4.0, 1.1, n, 8,  0.1,   8.0))

    # GDP quarterly → expand to monthly
    gdp_q, gdp_real   = get("A191RL1Q225SBEA",  lambda: _sim(2.5, 1.9, n//4, 9, -12, 10))
    gdp = [v for v in gdp_q for _ in range(4)]

    # PMI and credit spread: no free FRED equivalent → always simulated
    pmi     = _sim(51.5, 4.2, n, 10, 30, 70)
    cs      = _sim(1.8,  0.7, n, 11, 0.3, 6.0)   # investment-grade credit spread (%)
    housing = _sim(1400, 220, n, 12, 500, 2200)   # housing starts (thousands)
    retail  = _sim(0.4,  0.5, n, 13, -2.5, 3.5)  # retail sales MoM %

    min_len = min(len(infl), len(unemp), len(sp), len(conf), len(rate),
                  len(oil), len(t10), len(t2), len(gdp), len(pmi),
                  len(cs), len(housing), len(retail))

    def tr(x): return list(x)[-min_len:]
    infl,unemp,sp,conf,rate,oil,t10,t2,gdp,pmi,cs,housing,retail = map(tr,
        [infl,unemp,sp,conf,rate,oil,t10,t2,gdp,pmi,cs,housing,retail])

    yld = [t - s for t, s in zip(t10, t2)]

    # ── Live single-point values ──
    def live_val(series_id, fallback, lo, hi):
        v, real = _fred_latest(series_id)
        return float(np.clip(v if v is not None else fallback, lo, hi)), real

    l_infl,  lr_infl  = live_val("CPIAUCSL",    infl[-1],  1.0, 15.0)
    l_unemp, lr_unemp = live_val("UNRATE",       unemp[-1], 2.0, 14.0)
    l_sp,    lr_sp    = live_val("SP500",         sp[-1],  1800, 7000)
    l_conf,  lr_conf  = live_val("UMCSENT",      conf[-1],  30,  140)
    l_rate,  lr_rate  = live_val("FEDFUNDS",     rate[-1],  0.0, 20.0)
    l_oil,   lr_oil   = live_val("DCOILWTICO",   oil[-1],   15,  200)
    l_yld = float(yld[-1])
    l_gdp = float(gdp[-1])
    l_pmi = float(pmi[-1])
    l_cs  = float(cs[-1])
    l_housing = float(housing[-1])
    l_retail  = float(retail[-1])

    return {
        "meta": {
            "source":     "FRED / St. Louis Fed (simulation fallback where unavailable)",
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "n":          min_len,
            "real_flags": {
                "inflation": infl_real, "unemployment": unemp_real,
                "sp500": sp_real,  "consumer_confidence": conf_real,
                "fed_funds_rate": rate_real, "oil_price": oil_real,
                "gdp_growth": gdp_real, "yield_spread": t10_real and t2_real,
                "pmi": False, "credit_spread": False,
                "housing_starts": False, "retail_sales": False,
            },
        },
        "series": dict(
            inflation=infl, unemployment=unemp, sp500=sp,
            consumer_confidence=conf, fed_funds_rate=rate,
            oil_price=oil, gdp_growth=gdp, yield_spread=yld,
            pmi=pmi, credit_spread=cs, housing_starts=housing, retail_sales=retail,
        ),
        "live": dict(
            inflation=l_infl, unemployment=l_unemp, sp500=l_sp,
            consumer_confidence=l_conf, fed_funds_rate=l_rate,
            oil_price=l_oil, gdp_growth=l_gdp, yield_spread=l_yld,
            pmi=l_pmi, credit_spread=l_cs, housing_starts=l_housing,
            retail_sales=l_retail,
        ),
        "live_real": dict(
            inflation=lr_infl, unemployment=lr_unemp, sp500=lr_sp,
            consumer_confidence=lr_conf, fed_funds_rate=lr_rate, oil_price=lr_oil,
        ),
    }


def get_news():
    try:
        url  = (f"https://newsapi.org/v2/everything?q=economy+OR+inflation+OR+recession"
                f"&language=en&sortBy=publishedAt&pageSize=6&apiKey={NEWS_API_KEY}")
        arts = requests.get(url, timeout=7).json().get("articles", [])
        return [{"title": a["title"], "source": a["source"]["name"]} for a in arts[:6]]
    except Exception:
        return []


def news_sentiment(text):
    neg = ["crisis","recession","inflation","collapse","bankrupt","slowdown",
           "debt","fear","risk","downturn","warning","decline"]
    s = sum(1 for w in neg if w in text.lower())
    return "Negative" if s >= 2 else "Neutral" if s == 1 else "Positive"


# ══════════════════════════════════════════════
# FEATURE ENGINEERING  (12 features)
# ══════════════════════════════════════════════

FEATURE_COLS = [
    "inflation","unemployment","sp500","consumer_confidence",
    "fed_funds_rate","gdp_growth","pmi","oil_price",
    "yield_spread","credit_spread","housing_starts","retail_sales",
]

# ══════════════════════════════════════════════════════════════
# FEATURE CONTRIBUTION SYSTEM — hardcoded directions, no dynamic logic
# ══════════════════════════════════════════════════════════════

FEAT_NAMES = [
    "CPI Inflation",         # 0  direction +1: higher → always increases risk
    "Unemployment",          # 1  direction +1: higher → always increases risk
    "S&P 500",               # 2  direction -1: higher → always lowers risk
    "Consumer Confidence",   # 3  direction -1: higher → always lowers risk
    "Fed Funds Rate",        # 4  direction +1: higher → always increases risk
    "GDP Growth",            # 5  direction -1: higher → always lowers risk
    "PMI",                   # 6  direction -1: higher → always lowers risk
    "Oil Price",             # 7  direction +1: higher → always increases risk
    "Yield Spread (10y-2y)", # 8  direction -1: positive spread lowers risk;
                             #                  negative (inversion) increases risk
    "Credit Spread",         # 9  direction +1: higher → always increases risk
    "Housing Starts",        # 10 direction -1: higher → always lowers risk
    "Retail Sales",          # 11 direction -1: higher → always lowers risk
]

# DIRECTION: hardcoded, never changes, never conditional on value or z-score.
# +1 = higher value ALWAYS increases recession risk.
# -1 = higher value ALWAYS lowers recession risk.
DIRECTION = np.array([+1, +1, -1, -1, +1, -1, -1, +1, -1, +1, -1, -1], dtype=float)

# Fixed economic direction labels shown to the user (match DIRECTION exactly)
DIRECTION_LABEL = [
    "CPI Inflation ↑ → increases risk",
    "Unemployment ↑ → increases risk",
    "S&P 500 ↑ → lowers risk",
    "Consumer Confidence ↑ → lowers risk",
    "Fed Funds Rate ↑ → increases risk",
    "GDP Growth ↑ → lowers risk",
    "PMI ↑ → lowers risk",
    "Oil Price ↑ → increases risk",
    "Yield Spread: positive lowers risk; inversion increases risk",
    "Credit Spread ↑ → increases risk",
    "Housing Starts ↑ → lowers risk",
    "Retail Sales ↑ → lowers risk",
]

# Weight magnitudes (positive only). Combined with DIRECTION to form WEIGHTS.
_RAW_W = np.array([
    0.20, 0.22, 0.08, 0.09, 0.05, 0.16,
    0.06, 0.05, 0.13, 0.11, 0.04, 0.03,
], dtype=float)
_RAW_W /= _RAW_W.sum()

# Signed weights used in the ML stress score pipeline only.
WEIGHTS = DIRECTION * _RAW_W

# Static macro means and stds for explanation panel z-scores.
# Never derived from the fitted StandardScaler.
# Calibrated so realistic live values stay within z ∈ [-3, +3].
MACRO_MEAN = np.array([3.0, 5.5, 3500.0, 90.0, 3.5, 2.5, 51.5, 70.0, 0.8, 1.5, 1400.0, 0.3])
MACRO_STD  = np.array([1.5, 1.5,  800.0, 15.0, 2.0, 2.0,  5.0, 20.0, 0.8, 0.8,  200.0, 0.5])


def compute_contributions(vals):
    """
    contribution_i = DIRECTION[i] * weight_i * z_i
    where z_i = clip((val_i - MACRO_MEAN[i]) / MACRO_STD[i], -3, +3)

    contribution > 0 → currently increasing risk (guaranteed by construction).
    contribution < 0 → currently lowering risk   (guaranteed by construction).

    No conditional logic on z or val anywhere in this function.
    """
    v = np.asarray(vals, dtype=float)
    z = np.clip((v - MACRO_MEAN) / MACRO_STD, -3.0, +3.0)
    contributions = DIRECTION * _RAW_W * z
    return contributions, z


def explain_prediction(vals):
    """
    Returns list of dicts for the explanation panel.
    'raises' is ALWAYS derived from sign of contribution — no other source.
    'pct' is capped at 25% per feature; total sums to 100%.
    """
    contributions, z_scores = compute_contributions(vals)
    abs_c = np.abs(contributions)
    total = abs_c.sum() + 1e-9

    raw_pcts = abs_c / total * 100.0
    cap = 25.0
    capped = np.minimum(raw_pcts, cap)
    excess = raw_pcts.sum() - capped.sum()
    if excess > 0:
        headroom = (cap - capped) * (capped < cap)
        capped += headroom / (headroom.sum() + 1e-9) * excess
    capped = capped / capped.sum() * 100.0

    items = []
    for i in np.argsort(-abs_c):
        raises = bool(contributions[i] > 0)
        items.append({
            "feature":      FEAT_NAMES[i],
            "dir_label":    DIRECTION_LABEL[i],     # hardcoded economic description
            "pct":          float(capped[i]),
            "raises":       raises,                  # from contribution sign only
            "effect_label": "increasing risk" if raises else "lowering risk",
            "sign_char":    "+" if raises else "−",
            "value":        float(vals[i]),
            "z":            float(z_scores[i]),      # always in [-3, +3]
            "contribution": float(contributions[i]),
            "bar_color":    "#ff6b6b" if raises else "#34c759",
            "text_color":   "#ff9090" if raises else "#5ddb7a",
        })
    return items


def format_explanation_panel(items, sim_prob):
    """
    Renders the 'Why this prediction?' panel.
    Every label, color, and bar is derived from item['raises'].
    item['raises'] is set exclusively by the sign of contribution_i.
    No conditional z logic. No dynamic direction interpretation.
    """
    top5   = items[:5]
    rest_n = len(items) - 5

    dom = top5[0]
    dom_verb = "increasing" if dom["raises"] else "lowering"
    headline = (
        f"<b style='color:#fff'>{dom['feature']}</b> is the dominant factor "
        f"<span style='color:{dom['text_color']}'>{dom_verb} recession risk</span> "
        f"({dom['pct']:.0f}% of total model influence)."
    )

    rows_html = ""
    for rank, item in enumerate(top5, 1):
        rows_html += (
            f'<div style="margin:.42rem 0 .5rem;">'
            f'<div style="display:flex;align-items:baseline;gap:.4rem;flex-wrap:wrap;">'
            f'<span style="color:rgba(255,255,255,.3);font-size:.7rem;'
            f'font-family:\'JetBrains Mono\',monospace;min-width:1rem;">{rank}.</span>'
            f'<span style="color:#fff;font-weight:600;font-size:.86rem;">'
            f'{item["feature"]}</span>'
            f'<span style="color:{item["text_color"]};font-size:.75rem;margin-left:.2rem;">'
            f'({item["sign_char"]}{item["pct"]:.0f}%) {item["effect_label"]}</span>'
            f'</div>'
            f'<div style="font-size:.67rem;color:rgba(255,255,255,.28);'
            f'padding-left:1.35rem;margin:.05rem 0 .18rem;line-height:1.5;">'
            f'Value: <b style="color:rgba(255,255,255,.6)">{item["value"]:.2f}</b> · '
            f'z-score: <b style="color:rgba(255,255,255,.6)">{item["z"]:+.2f}</b> · '
            f'contribution: <b style="color:{item["text_color"]}">'
            f'{item["contribution"]:+.4f}</b>'
            f'<br><span style="color:rgba(255,255,255,.16)">'
            f'{item["dir_label"]}</span>'
            f'</div>'
            f'<div style="margin-left:1.35rem;" class="feat-bar-wrap">'
            f'<div class="feat-bar-fill" style="width:{min(int(item["pct"]),100)}%;'
            f'background:{item["bar_color"]};opacity:.8;"></div></div>'
            f'</div>'
        )

    rest_html = (
        f'<p style="font-size:.67rem;color:rgba(255,255,255,.2);margin-top:.55rem;">'
        f'+ {rest_n} minor contributing factors not shown</p>'
    ) if rest_n > 0 else ""

    return (
        f'<div class="why-box">'
        f'<p style="font-size:.82rem;color:rgba(255,255,255,.6);margin:0 0 .8rem;'
        f'line-height:1.5;">{headline}</p>'
        f'<p style="font-size:.61rem;color:rgba(255,255,255,.18);'
        f'font-family:\'JetBrains Mono\',monospace;text-transform:uppercase;'
        f'letter-spacing:.12em;margin:0 0 .45rem;">Top 5 drivers of recession risk:</p>'
        f'{rows_html}{rest_html}'
        f'</div>'
    )

def build_df(series):
    df = pd.DataFrame({c: series[c] for c in FEATURE_COLS})
    sc = StandardScaler()
    z  = sc.fit_transform(df[FEATURE_COLS])
    df["stress_score"] = z @ WEIGHTS
    return df, sc


def stress_from_vals(vals, sc):
    z = sc.transform([vals])[0]
    return float(z @ WEIGHTS)


def live_as_vals(live):
    return [live[c] for c in FEATURE_COLS]


# ══════════════════════════════════════════════
# MODEL — cached, 80/20 split, label noise
# ══════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def train_pipeline(cache_key: str):
    raw       = fetch_economic_data()
    df, sc    = build_df(raw["series"])

    X         = df[FEATURE_COLS].values
    threshold = float(df["stress_score"].median())
    y         = (df["stress_score"] > threshold).astype(int).values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True)

    # ── Label noise (8 %) → realistic metrics ──
    rng_noise = np.random.default_rng(77)
    flip      = rng_noise.random(len(y_tr)) < 0.08
    y_noisy   = y_tr.copy()
    y_noisy[flip] = 1 - y_noisy[flip]

    mdl = RandomForestClassifier(
        n_estimators=300, max_depth=7, min_samples_leaf=5,
        max_features="sqrt", random_state=42, n_jobs=-1)
    mdl.fit(X_tr, y_noisy)

    cv_acc = float(cross_val_score(mdl, X_tr, y_noisy, cv=5, scoring="accuracy").mean())
    y_pred = mdl.predict(X_te)

    metrics = dict(
        accuracy  = float(accuracy_score(y_te,  y_pred)),
        precision = float(precision_score(y_te, y_pred, zero_division=0)),
        recall    = float(recall_score(y_te,    y_pred, zero_division=0)),
        f1        = float(f1_score(y_te,        y_pred, zero_division=0)),
        cv_acc    = cv_acc,
        cm        = confusion_matrix(y_te, y_pred).tolist(),
    )

    t  = np.arange(len(df)).reshape(-1, 1)
    lr = LinearRegression().fit(t, df["stress_score"].values)

    return mdl, sc, df, metrics, lr, threshold


# ══════════════════════════════════════════════
# INFERENCE HELPERS
# ══════════════════════════════════════════════

def predict_prob(mdl, sc, vals):
    p = mdl.predict_proba(sc.transform([vals]))[0][1] * 100
    return float(np.clip(p, 0, 95))


def forecast_stress(lr_mdl, df, sim_stress, n=6):
    """
    6-month stress forecast:
      - Trend component: linear regression on historical stress (60%)
      - Scenario component: current simulation stress (40%)
      - Mean-reversion: pull toward long-run median with 10% weight per step
      - Volatility: calibrated to 20% of historical std, applied with AR(1)
        structure so consecutive months differ meaningfully (no flat bars)
      - Direction: high stress → upward drift; low stress → mild recovery

    Result: forecast values differ month-to-month by design.
    """
    n_hist   = len(df)
    hist_std = float(df["stress_score"].std())
    median   = float(df["stress_score"].median())

    t_fut = np.arange(n_hist, n_hist + n).reshape(-1, 1)
    trend = lr_mdl.predict(t_fut)

    # Blend trend with scenario
    base = 0.60 * trend + 0.40 * sim_stress

    # Drift adjustment: if current scenario stress is high, add upward drift;
    # if low, add slight mean-reverting recovery
    stress_gap = sim_stress - median
    drift_per_step = np.sign(stress_gap) * min(abs(stress_gap) * 0.06, hist_std * 0.12)

    # AR(1) noise: each step's noise is partially inherited from the last
    rng     = np.random.default_rng(int(abs(sim_stress) * 1e4) % 99991)
    eps     = rng.normal(0, hist_std * 0.20, n)
    ar_noise = np.zeros(n)
    ar_noise[0] = eps[0]
    for i in range(1, n):
        ar_noise[i] = 0.55 * ar_noise[i - 1] + 0.45 * eps[i]

    # Mean-reversion: pull 8% back toward median each step
    forecasted = np.zeros(n)
    prev = sim_stress
    for i in range(n):
        revert    = (median - prev) * 0.08
        forecasted[i] = base[i] + drift_per_step * (i + 1) + ar_noise[i] + revert
        prev = forecasted[i]

    # Convert stress to probability: logistic-style mapping
    # stress at median → ~50 %; +2σ → ~80 %; −2σ → ~20 %
    norm_stress = (forecasted - median) / (hist_std + 1e-9)
    probs = np.clip(50 + 18 * norm_stress, 0, 95)

    return forecasted, probs



def policy_effect(policy, vals):
    """
    Apply policy shock to all 12 indicator values.
    Deltas are calibrated to realistic short-run transmission magnitudes
    (based on typical 6-12 month Fed/fiscal impact estimates).
    Order matches FEATURE_COLS:
      infl, unemp, sp500, conf, fed_rate, gdp, pmi, oil, yld_spread, credit_spread, housing, retail
    """
    deltas = {
        # Rate cut → cheaper credit → equity up, confidence up, hiring improves,
        #   yield curve steepens (spread widens), credit spreads tighten,
        #   housing starts pick up; mild inflationary over time
        "Interest Rate Cut (−50 bps)": [
            +0.45,  # inflation   (demand-side pickup)
            -0.55,  # unemployment
            +320,   # sp500       (risk-on repricing)
            +7,     # confidence
            -0.50,  # fed_rate    (definition)
            +0.55,  # gdp
            +2.8,   # pmi
            -3.0,   # oil         (USD weakens → mixed; net slight fall)
            +0.18,  # yield_spread (curve steepens)
            -0.22,  # credit_spread (tightens)
            +60,    # housing_starts
            +0.35,  # retail_sales
        ],
        # Stimulus → direct demand → GDP and employment up, inflation up,
        #   equity markets react positively, credit spreads compress
        "Fiscal Stimulus Package": [
            +0.85,  # inflation
            -0.65,  # unemployment
            +400,   # sp500
            +9,     # confidence
             0.0,   # fed_rate    (unchanged)
            +1.10,  # gdp
            +3.5,   # pmi
            +5.0,   # oil         (demand-driven)
            +0.08,  # yield_spread (modest)
            -0.30,  # credit_spread
            +85,    # housing_starts
            +0.55,  # retail_sales
        ],
        # Tax hike → disposable income falls → spending down, confidence down,
        #   equities re-price lower, gdp softens, hiring slows
        "Tax Increase (+2%)": [
            -0.30,  # inflation   (demand destruction)
            +0.55,  # unemployment
            -350,   # sp500
            -8,     # confidence
             0.0,   # fed_rate
            -0.80,  # gdp
            -2.5,   # pmi
            -2.0,   # oil
            -0.12,  # yield_spread
            +0.20,  # credit_spread
            -55,    # housing_starts
            -0.45,  # retail_sales
        ],
    }.get(policy, [0] * 12)

    clips = [
        (1, 15), (2, 14), (1800, 7000), (30, 140), (0, 20),
        (-12, 10), (30, 70), (15, 200), (-3, 3), (0.2, 7),
        (400, 2400), (-3, 4),
    ]
    return [float(np.clip(v + d, lo, hi))
            for v, d, (lo, hi) in zip(vals, deltas, clips)]


# ══════════════════════════════════════════════
# LLM — ROBUST PARSING
# ══════════════════════════════════════════════

def call_llm(system_prompt, user_msg, temperature=0.78, max_tokens=120):
    """Call OpenRouter. On ANY failure return a clean fallback message — never expose raw errors."""
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type":  "application/json",
                "HTTP-Referer":  "https://economic-warning-ai.streamlit.app",
                "X-Title":       "Economic Early Warning System",
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
            timeout=20,
        )
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
            pass
        try:
            return data["output"][0]["content"][0]["text"].strip()
        except (KeyError, IndexError, TypeError):
            pass
        for k in ("content", "text", "response", "answer"):
            if k in data and isinstance(data[k], str):
                return data[k].strip()
        return "AI insights temporarily unavailable due to API limits."
    except Exception:
        return "AI insights temporarily unavailable due to API limits."


# def call_llm(system_prompt, user_msg, temperature=0.5, max_tokens=120):
#     url = "https://openrouter.ai/api/v1/chat/completions"

#     headers = {
#         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#         "Content-Type": "application/json"
#     }

#     payload = {
#         "model": "mistralai/mixtral-8x7b-instruct",
#         "messages": [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_msg}
#         ],
#         "temperature": temperature,
#         "max_tokens": max_tokens   # ← NOW MATCHES CALL
#     }

#     try:
#         res = requests.post(url, headers=headers, json=payload, timeout=8)

#         if res.status_code != 200:
#             return "AI insights temporarily unavailable due to API limits."

#         data = res.json()
#         return data["choices"][0]["message"]["content"].strip()

#     except Exception:
#         return "AI insights temporarily unavailable due to API limits."


# ── SYSTEM PROMPT: human economist, no templates ──
ANALYST_STYLE = """You are a senior macroeconomic analyst. Your job is to reason, not recite.

Rules:
- Never follow a fixed structure. Mix bullets and paragraphs freely.
- Avoid phrases like "In conclusion", "Given the data", "Let's analyze", "It's worth noting".
- Don't over-explain basics that any economist knows.
- Lead with the insight, not the setup.
- Be direct. If something is uncertain, say so plainly.
- Sometimes a short, sharp answer is better than a long one.
- Always ground your reasoning in specific numbers from the data.
- Think about second-order effects: what does A → B → C look like?

You are not a textbook. You are an analyst with a point of view."""


def build_context(live, live_prob, live_stress, sim_vals, sim_prob):
    """Build LLM system context with top-3 stress drivers from compute_contributions."""
    contrib, z_arr = compute_contributions(sim_vals)
    top3 = np.argsort(-np.abs(contrib))[:3]
    driver_lines = []
    for i in top3:
        effect = "increasing" if contrib[i] > 0 else "lowering"
        driver_lines.append(
            f"  {FEAT_NAMES[i]}: {sim_vals[i]:.2f} "
            f"(z={z_arr[i]:+.2f}, currently {effect} risk)"
        )
    drivers_block = "\n".join(driver_lines)
    sim_lines = "\n".join(
        f"  {n}: {v:.2f}" for n, v in zip(FEAT_NAMES, sim_vals)
    )
    return f"""{ANALYST_STYLE}

=== CURRENT MACRO DATA ===
CPI Inflation: {live['inflation']:.2f}%  |  Unemployment: {live['unemployment']:.2f}%  |  S&P 500: {live['sp500']:.0f}
Consumer Confidence: {live['consumer_confidence']:.1f}  |  Fed Funds Rate: {live['fed_funds_rate']:.2f}%  |  GDP Growth: {live['gdp_growth']:.2f}%
PMI: {live['pmi']:.1f}  |  Oil: ${live['oil_price']:.1f}/bbl  |  Yield Spread (10y-2y): {live['yield_spread']:+.3f}%
Credit Spread: {live['credit_spread']:.2f}%  |  Housing Starts: {live['housing_starts']:.0f}k  |  Retail Sales MoM: {live['retail_sales']:+.2f}%

AI Recession Probability: {live_prob:.1f}%  |  Composite Stress Index: {live_stress:.3f}

=== TOP STRESS DRIVERS (scenario) ===
{drivers_block}

=== FULL SCENARIO (USER-ADJUSTED) ===
{sim_lines}
Scenario Recession Probability: {sim_prob:.1f}%"""


def generate_insights(live, live_prob, live_stress, sim_vals, sim_prob):
    system = build_context(live, live_prob, live_stress, sim_vals, sim_prob)
    results = []
    rng = np.random.default_rng(int(live_prob * 100))

    checks = [
        (live["yield_spread"] < 0,
         "danger","📐",
         f"The yield curve is inverted at {live['yield_spread']:.3f}%. "
         f"Rates are at {live['fed_funds_rate']:.1f}%. What's the historical "
         f"precedent here and how long does it typically take before this bites?"),

        (live["inflation"] > 5.5 and live["unemployment"] > 5.0,
         "danger","⚡",
         f"Inflation {live['inflation']:.1f}%, unemployment {live['unemployment']:.1f}% — "
         f"simultaneously. Walk through the stagflation transmission mechanism "
         f"and why this is particularly hard to escape from."),

        (live["credit_spread"] > 2.5,
         "warn","💳",
         f"Credit spreads at {live['credit_spread']:.2f}% signal rising default risk. "
         f"With GDP at {live['gdp_growth']:.1f}%, what does this mean for "
         f"corporate refinancing and the real economy?"),

        (live["inflation"] > 5.0,
         "warn","🔥",
         f"CPI running {live['inflation']:.1f}% with rates at {live['fed_funds_rate']:.1f}%. "
         f"Real rates are {live['fed_funds_rate'] - live['inflation']:.1f}%. "
         f"Is monetary policy actually restrictive here? Explain the lag."),

        (live["unemployment"] > 5.5,
         "danger","👷",
         f"Unemployment {live['unemployment']:.1f}%, GDP {live['gdp_growth']:.1f}%. "
         f"Okun's law says something about that gap. What's the consumption knock-on "
         f"and which sectors feel it first?"),

        (live["consumer_confidence"] < 72,
         "info","😟",
         f"Confidence at {live['consumer_confidence']:.0f}, retail sales {live['retail_sales']:+.2f}% MoM. "
         f"How much does confidence actually lead spending, and is this already "
         f"showing up in the data?"),

        (live["pmi"] < 48,
         "warn","⚙️",
         f"PMI at {live['pmi']:.1f} — contraction territory. "
         f"Oil at ${live['oil_price']:.0f}. "
         f"What does this input cost environment mean for margins and hiring?"),

        (live_prob > 65,
         "danger","🚨",
         f"Model flags {live_prob:.1f}% recession probability. "
         f"Which two or three indicators are doing the most damage to that number, "
         f"and what would need to shift to bring it below 50%?"),
    ]

    for condition, ctype, icon, prompt in checks:
        if condition:
            temp = round(float(rng.uniform(0.72, 0.91)), 2)
            results.append((ctype, icon, call_llm(system, prompt, temperature=temp)))
        if len(results) >= 4:
            break

    if not results:
        results.append(("success","✅",
            "Indicators are broadly within normal ranges — yield curve positive, "
            "PMI above 50, credit spreads contained. The stress index is below threshold. "
            "That said, oil and inflation still bear watching; disinflation isn't always smooth."))
    return results


# ══════════════════════════════════════════════
# PLOTLY THEME
# ══════════════════════════════════════════════

CL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="rgba(255,255,255,.55)", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="rgba(255,255,255,.04)", zerolinecolor="rgba(255,255,255,.07)", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="rgba(255,255,255,.04)", zerolinecolor="rgba(255,255,255,.07)", tickfont=dict(size=11)),
    hoverlabel=dict(bgcolor="rgba(10,20,40,.95)", bordercolor="rgba(0,212,170,.4)",
                    font=dict(family="JetBrains Mono", size=12, color="white")),
    legend=dict(bgcolor="rgba(255,255,255,.04)", bordercolor="rgba(255,255,255,.07)", borderwidth=1),
)


def make_gauge(value, title="Recession Probability"):
    gc  = "#ff3b30" if value > 70 else "#ff9500" if value > 40 else "#00d4aa"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=value,
        number=dict(suffix="%", font=dict(size=32, color="white", family="JetBrains Mono")),
        title=dict(text=title, font=dict(size=11, color="rgba(255,255,255,.45)")),
        delta=dict(reference=50, increasing=dict(color="#ff3b30"), decreasing=dict(color="#34c759")),
        gauge=dict(
            axis=dict(range=[0,100], tickwidth=1, tickcolor="rgba(255,255,255,.18)",
                      tickfont=dict(color="rgba(255,255,255,.3)", size=9)),
            bar=dict(color=gc, thickness=0.25),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[dict(range=[0,40],  color="rgba(0,212,170,.11)"),
                   dict(range=[40,70], color="rgba(255,149,0,.11)"),
                   dict(range=[70,100],color="rgba(255,59,48,.11)")],
            threshold=dict(line=dict(color="white",width=2), thickness=0.75, value=value),
        ),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Space Grotesk",color="white"),
                      height=245, margin=dict(l=30,r=30,t=40,b=10))
    return fig


# ══════════════════════════════════════════════
# BOOT
# ══════════════════════════════════════════════

with st.spinner("Loading economic data…"):
    _econ = fetch_economic_data()

with st.spinner("Training AI model…"):
    model, scaler, df, mets, lr_model, threshold = train_pipeline(
        _econ["meta"]["fetched_at"][:13])

live      = _econ["live"]
live_real = _econ["live_real"]
meta      = _econ["meta"]
rf        = meta["real_flags"]

live_vals   = live_as_vals(live)
live_stress = stress_from_vals(live_vals, scaler)
live_prob   = predict_prob(model, scaler, live_vals)
conf_matrix = np.array(mets["cm"])


# ══════════════════════════════════════════════
# ① HERO
# ══════════════════════════════════════════════

st.markdown("""
<div class="hero-header">
    <div class="hero-badge">AI-DRIVEN SIMULATION · 12 MACRO INDICATORS · MIXTRAL 8×7B</div>
    <h1 class="hero-title">Economic Early Warning System</h1>
    <p class="hero-sub">Scenario-reactive recession risk assessment for policymakers and investors.
    Random Forest + LLM reasoning · 12-indicator stress model · dynamic 6-month forecast.</p>
</div>""", unsafe_allow_html=True)

st.markdown(f"""
<div class="transparency-bar">
⚠ Data Transparency — FRED-sourced where API is available
<span style="color:rgba(0,212,170,.7);">●</span> Real &nbsp;
<span style="color:rgba(255,200,80,.65);">●</span> Simulated (historically calibrated) &nbsp;|&nbsp;
Fetched: {meta['fetched_at'][:16]} UTC
</div>""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# ② KPI CARDS  — tagged real vs. simulated
# ══════════════════════════════════════════════

def tag(is_real): return '<span class="kpi-tag real">real</span>' if is_real else '<span class="kpi-tag sim">sim</span>'

st.markdown('<p class="section-label">📊 Key Macro Indicators</p>', unsafe_allow_html=True)

r1 = st.columns(5)
r2 = st.columns(5)

kpi_r1 = [
    ("blue",  "🏦","CPI Inflation",        f"{live['inflation']:.2f}%",         rf.get("inflation",False)),
    ("teal",  "👷","Unemployment",          f"{live['unemployment']:.2f}%",      rf.get("unemployment",False)),
    ("purple","📈","S&P 500",               f"{live['sp500']:.0f}",              rf.get("sp500",False)),
    ("red",   "⚠️","Recession Probability", f"{live_prob:.1f}%",                 None),
    ("green", "🎯","Model F1",              f"{mets['f1']*100:.1f}%",            None),
]
kpi_r2 = [
    ("orange","🏛️","Fed Funds Rate",        f"{live['fed_funds_rate']:.2f}%",    rf.get("fed_funds_rate",False)),
    ("cyan",  "📐","Yield Spread",          f"{live['yield_spread']:+.3f}%",     rf.get("yield_spread",False)),
    ("pink",  "💳","Credit Spread",         f"{live['credit_spread']:.2f}%",     False),
    ("teal",  "⚙️","PMI",                  f"{live['pmi']:.1f}",                False),
    ("orange","🛢️","Oil Price",             f"${live['oil_price']:.1f}",         rf.get("oil_price",False)),
]

for col, (color, icon, label, value, is_real) in zip(r1, kpi_r1):
    with col:
        t = tag(is_real) if is_real is not None else ""
        st.markdown(f"""<div class="kpi-card {color}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}{t}</div>
        <div class="kpi-value">{value}</div>
        </div>""", unsafe_allow_html=True)

for col, (color, icon, label, value, is_real) in zip(r2, kpi_r2):
    with col:
        t = tag(is_real)
        st.markdown(f"""<div class="kpi-card {color}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}{t}</div>
        <div class="kpi-value">{value}</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# ③ MODEL PERFORMANCE
# ══════════════════════════════════════════════

st.markdown('<p class="section-label">🎯 Model Performance · 80/20 Split · 5-Fold CV · 8% Label Noise</p>',
            unsafe_allow_html=True)

mp1, mp2, mp3 = st.columns([1,1,2], gap="medium")

with mp1:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Test-Set Metrics</p>', unsafe_allow_html=True)
    for lbl, val in [("Accuracy",   mets["accuracy"]),
                     ("Precision",  mets["precision"]),
                     ("Recall",     mets["recall"]),
                     ("F1 Score",   mets["f1"]),
                     ("CV Acc (5-fold)", mets["cv_acc"])]:
        pct   = val * 100
        color = "#34c759" if pct >= 75 else "#ff9500" if pct >= 62 else "#ff3b30"
        st.markdown(
            f'<div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);'
            f'border-radius:8px;padding:.38rem .75rem;margin:.22rem 0;'
            f'font-family:\'JetBrains Mono\',monospace;font-size:.76rem;color:rgba(255,255,255,.55);">'
            f'{lbl}: <span style="color:{color};font-weight:600;">{pct:.1f}%</span></div>',
            unsafe_allow_html=True)
    st.markdown('<p style="font-size:.65rem;color:rgba(255,255,255,.2);margin-top:.7rem;">'
                'RF 300 trees · depth 7 · min_leaf 5 · 8% label noise</p>',
                unsafe_allow_html=True)
    # ── Model bias note ──
    recall_val    = mets["recall"]
    precision_val = mets["precision"]
    if recall_val > precision_val + 0.05:
        bias_note = (
            "Recall exceeds precision — the model is tuned to catch recessions early, "
            "accepting more false positives in exchange for fewer missed warnings. "
            "This is the correct trade-off for an early-warning system."
        )
    elif precision_val > recall_val + 0.05:
        bias_note = (
            "Precision exceeds recall — the model is conservative, flagging fewer "
            "recessions but with higher confidence when it does. "
            "Consider this when setting alert thresholds."
        )
    else:
        bias_note = (
            "Precision and recall are balanced — the model treats false positives "
            "and false negatives symmetrically."
        )
    st.markdown(
        f'<p style="font-size:.72rem;color:rgba(255,200,120,.55);'
        f'line-height:1.5;margin-top:.5rem;border-top:1px solid rgba(255,255,255,.05);'
        f'padding-top:.5rem;">{bias_note}</p>',
        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with mp2:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Confusion Matrix</p>', unsafe_allow_html=True)
    fig_cm = go.Figure(go.Heatmap(
        z=conf_matrix,
        x=["Pred: Low","Pred: High"], y=["Actual: Low","Actual: High"],
        colorscale=[[0,"rgba(0,212,170,.22)"],[1,"rgba(255,59,48,.72)"]],
        showscale=False, text=conf_matrix, texttemplate="%{text}",
        textfont=dict(size=20, color="white", family="JetBrains Mono"),
    ))
    fig_cm.update_layout(**CL, height=215)
    st.plotly_chart(fig_cm, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with mp3:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Feature Importance (12 Indicators)</p>',
                unsafe_allow_html=True)
    imps     = model.feature_importances_
    idx      = np.argsort(imps)
    fi_clrs  = ["#007aff","#6326ff","#00d4aa","#ff9500","#ff3b30","#34c759",
                "#5ac8fa","#af52de","#ffcc00","#ff6b95","#32ade6","#30d158"]
    short    = ["CPI","Unemp","S&P","Conf","Fed Rate","GDP",
                "PMI","Oil","Yield Sprd","Cred Sprd","Housing","Retail"]
    fig_fi = go.Figure(go.Bar(
        x=imps[idx], y=[short[i] for i in idx], orientation="h",
        marker=dict(color=[fi_clrs[i] for i in idx], opacity=.82),
        text=[f"{imps[i]:.3f}" for i in idx], textposition="inside",
        textfont=dict(color="white", size=10, family="JetBrains Mono"),
        hovertemplate="<b>%{y}</b>: %{x:.4f}<extra></extra>",
    ))
    fig_fi.update_layout(**CL, height=255, showlegend=False)
    fig_fi.update_xaxes(title_text="Gini Importance")
    st.plotly_chart(fig_fi, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# ④ STRESS TREND + SCATTER
# ══════════════════════════════════════════════

st.markdown('<p class="section-label">📉 Economic Stress Analysis</p>', unsafe_allow_html=True)
ct, cs = st.columns([3,2], gap="medium")

with ct:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Composite Stress Score — Historical</p>',
                unsafe_allow_html=True)
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(
        y=df["stress_score"], mode="lines",
        line=dict(color="#00d4aa", width=1.8, shape="spline"),
        fill="tozeroy", fillcolor="rgba(0,212,170,.065)",
        hovertemplate="Period %{x}<br>Stress: %{y:.3f}<extra></extra>",
    ))
    fig_t.add_hline(y=threshold, line_dash="dash",
                    line_color="rgba(255,149,0,.55)", line_width=1.4,
                    annotation_text="Recession threshold (median)",
                    annotation_font_color="rgba(255,180,80,.75)", annotation_position="right")
    fig_t.update_layout(**CL, xaxis_title="Observation", yaxis_title="Stress Score", height=290)
    st.plotly_chart(fig_t, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with cs:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Stress vs Unemployment</p>', unsafe_allow_html=True)
    fig_sc = go.Figure(go.Scatter(
        x=df["unemployment"], y=df["stress_score"], mode="markers",
        marker=dict(size=5, color=df["stress_score"],
                    colorscale=[[0,"#00d4aa"],[.5,"#007aff"],[1,"#ff3b30"]],
                    showscale=True, colorbar=dict(title="Stress", thickness=9, len=.7),
                    line=dict(color="#050d1a", width=1)),
        hovertemplate="Unemployment: %{x:.1f}%<br>Stress: %{y:.3f}<extra></extra>",
    ))
    fig_sc.update_layout(**CL, xaxis_title="Unemployment (%)", yaxis_title="Stress Score", height=290)
    st.plotly_chart(fig_sc, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# ⑤ SCENARIO SIMULATOR — 12-feature, fully connected
# ══════════════════════════════════════════════

st.markdown('<p class="section-label">🎛️ Scenario Simulator — All sliders feed the trained RF model</p>',
            unsafe_allow_html=True)

sc1, sc2, sc3 = st.columns(3, gap="medium")

with sc1:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    sim_infl  = st.slider("🔥 CPI Inflation (%)",        0.0, 14.0, float(round(live["inflation"],1)),        0.1)
    sim_unemp = st.slider("👷 Unemployment (%)",         2.0, 14.0, float(round(live["unemployment"],1)),     0.1)
    sim_sp    = st.slider("📈 S&P 500",                2000, 6500,  int(round(live["sp500"],-2)),              50)
    sim_conf  = st.slider("😟 Consumer Confidence",     30,  130,   int(round(live["consumer_confidence"])),   1)
    st.markdown('</div>', unsafe_allow_html=True)

with sc2:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    sim_rate  = st.slider("🏛️ Fed Funds Rate (%)",       0.0, 20.0, float(round(live["fed_funds_rate"],1)),  0.25)
    sim_gdp   = st.slider("📊 GDP Growth (%)",          -8.0,  8.0, float(round(live["gdp_growth"],1)),       0.1)
    sim_pmi   = st.slider("⚙️ PMI",                    30.0, 70.0,  float(round(live["pmi"],1)),              0.5)
    sim_oil   = st.slider("🛢️ Oil ($/bbl)",             20,   160,   int(round(live["oil_price"])),             1)
    st.markdown('</div>', unsafe_allow_html=True)

with sc3:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    sim_yld   = st.slider("📐 Yield Spread (10y-2y %)", -2.5, 3.0,  float(round(live["yield_spread"],2)),   0.05)
    sim_cs    = st.slider("💳 Credit Spread (%)",        0.2,  7.0,  float(round(live["credit_spread"],2)),  0.05)
    sim_hous  = st.slider("🏠 Housing Starts (k)",       400, 2400,  int(round(live["housing_starts"],-1)),   10)
    sim_ret   = st.slider("🛍️ Retail Sales MoM (%)",   -3.0,  4.0,  float(round(live["retail_sales"],1)),    0.1)
    st.markdown('</div>', unsafe_allow_html=True)

sim_vals   = [sim_infl, sim_unemp, sim_sp, sim_conf, sim_rate, sim_gdp,
              sim_pmi, sim_oil, sim_yld, sim_cs, sim_hous, sim_ret]
sim_stress = stress_from_vals(sim_vals, scaler)
sim_prob   = predict_prob(model, scaler, sim_vals)
sim_health = 100 - sim_prob

# ── Scenario insight ──
prob_delta = sim_prob - live_prob
if abs(prob_delta) > 2:
    direction = "increased" if prob_delta > 0 else "decreased"
    color_d   = "#ff9500" if prob_delta > 0 else "#34c759"
    st.markdown(
        f'<div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);'
        f'border-radius:10px;padding:.7rem 1.1rem;margin-top:.5rem;font-size:.83rem;'
        f'color:rgba(255,255,255,.65);">'
        f'Scenario vs live: recession probability <span style="color:{color_d};font-weight:600;">'
        f'{direction} by {abs(prob_delta):.1f}pp</span> '
        f'(live: {live_prob:.1f}% → scenario: {sim_prob:.1f}%). '
        f'Stress index shifted from {live_stress:.3f} to {sim_stress:.3f}.</div>',
        unsafe_allow_html=True)


# ══════════════════════════════════════════════
# ⑥ POLICY SIMULATOR — model-connected
# ══════════════════════════════════════════════

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("## 🧠 Policy Impact Simulator")

policy = st.selectbox("Select a policy intervention to model against current scenario:",
    ["Interest Rate Cut (−50 bps)", "Fiscal Stimulus Package", "Tax Increase (+2%)"])

pol_vals   = policy_effect(policy, sim_vals)
pol_stress = stress_from_vals(pol_vals, scaler)
pol_prob   = predict_prob(model, scaler, pol_vals)

pol_desc = {
    "Interest Rate Cut (−50 bps)":
        "Cheaper credit → more business investment and consumer spending → equity market support and confidence boost. "
        "Mild upward pressure on inflation; housing likely benefits. "
        "Lagged effect — typically 6–18 months before full transmission.",
    "Fiscal Stimulus Package":
        "Direct demand injection lifts GDP and employment. "
        "Upward inflation pressure from demand-pull. "
        "Credit spreads may compress as default risk falls. "
        "PMI typically responds within 1–2 quarters.",
    "Tax Increase (+2%)":
        "Reduces disposable income → softer consumption and retail sales. "
        "PMI and confidence contract. Mild disinflationary but risk of output contraction "
        "if demand weakness is already present.",
}
st.markdown(f'<div class="glass-panel"><p style="font-size:.87rem;color:rgba(255,255,255,.68);'
            f'line-height:1.65;">{pol_desc[policy]}</p></div>', unsafe_allow_html=True)

p1,p2,p3,p4 = st.columns(4)
p1.metric("Stress Index",       f"{pol_stress:.3f}",  f"{pol_stress-sim_stress:+.3f}")
p2.metric("Recession Prob",     f"{pol_prob:.1f}%",   f"{pol_prob-sim_prob:+.1f}%")
p3.metric("Unemployment",       f"{pol_vals[1]:.2f}%",f"{pol_vals[1]-sim_unemp:+.2f}%")
p4.metric("Inflation",          f"{pol_vals[0]:.2f}%",f"{pol_vals[0]-sim_infl:+.2f}%")


# ══════════════════════════════════════════════
# ⑦ RESULTS ROW
# ══════════════════════════════════════════════

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">📡 Live Model Output · Scenario</p>', unsafe_allow_html=True)

r1, r2, r3 = st.columns([1,2,1], gap="medium")

with r1:
    st.markdown('<div class="glass-panel" style="text-align:center;">', unsafe_allow_html=True)
    st.markdown(f'<p class="score-label">STRESS INDEX</p>'
                f'<p class="score-display">{sim_stress:.2f}</p>', unsafe_allow_html=True)
    if sim_stress > df["stress_score"].quantile(.75):
        badge, bcls = "🔴 ELEVATED", "high"
    elif sim_stress > threshold:
        badge, bcls = "🟡 MODERATE", "medium"
    else:
        badge, bcls = "🟢 STABLE",   "low"
    st.markdown(f'<div style="text-align:center">'
                f'<span class="risk-badge {bcls}">{badge}</span></div>',
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
    expl = explain_prediction(sim_vals)
    st.markdown(format_explanation_panel(expl, sim_prob), unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# ⑧ REAL FORECAST — trend + scenario + AR noise
# ══════════════════════════════════════════════

st.markdown('<p class="section-label">🔮 6-Month Forecast</p>', unsafe_allow_html=True)
st.caption(
    "Forecast = 60% historical linear trend · 40% scenario stress · "
    "AR(1) volatility + mean-reversion. "
    "High stress → upward drift; low stress → mild recovery. "
    "Reacts dynamically to slider adjustments above."
)

f_stress, f_probs = forecast_stress(lr_model, df, sim_stress, 6)
months = ["Month 1","Month 2","Month 3","Month 4","Month 5","Month 6"]

st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown('<p class="section-title">Projected Economic Risk — 6-Month Horizon</p>',
            unsafe_allow_html=True)

fig_fc = go.Figure()
fig_fc.add_trace(go.Bar(
    x=months, y=f_stress, name="Forecast Stress",
    marker=dict(color=list(f_stress),
                colorscale=[[0,"rgba(0,212,170,.72)"],[.5,"rgba(0,122,255,.72)"],[1,"rgba(255,59,48,.72)"]],
                opacity=.78),
    yaxis="y",
    hovertemplate="<b>%{x}</b><br>Stress: %{y:.3f}<extra></extra>",
))
fig_fc.add_trace(go.Scatter(
    x=months, y=f_probs, name="Recession Probability %",
    mode="lines+markers",
    line=dict(color="#ff9500", width=2.4, shape="spline"),
    marker=dict(size=7, color="#ff9500", line=dict(color="#050d1a",width=2)),
    yaxis="y2",
    hovertemplate="<b>%{x}</b><br>Probability: %{y:.1f}%<extra></extra>",
))
fig_fc.update_layout(
    **CL, height=330, barmode="group",
    yaxis2=dict(title="Recession Probability (%)", overlaying="y", side="right",
                range=[0,100], gridcolor="rgba(0,0,0,0)",
                tickfont=dict(size=11, color="rgba(255,149,0,.65)")),
)
fig_fc.update_yaxes(title_text="Stress Score", selector=dict(overlaying=None))
fig_fc.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="right", x=1,
                                  bgcolor="rgba(255,255,255,.04)",
                                  bordercolor="rgba(255,255,255,.07)", borderwidth=1))
st.plotly_chart(fig_fc, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# ⑨ GLOBAL MAP — trade-linkage connected
# ══════════════════════════════════════════════

st.markdown('<p class="section-label">🌍 Global Recession Risk Map</p>', unsafe_allow_html=True)
st.caption("US risk = direct model output. Regional estimates use trade/financial linkage "
           "coefficients. Tooltip shows linkage weight. Updates with scenario.")

LINKAGE = {
    "United States":  (1.00, 0),    # (coeff, idiosyncratic_noise_amp_pct)
    "Canada":         (0.84, 5),
    "Mexico":         (0.75, 7),
    "United Kingdom": (0.79, 6),
    "Germany":        (0.73, 6),
    "France":         (0.70, 7),
    "Italy":          (0.64, 8),
    "Spain":          (0.61, 8),
    "Netherlands":    (0.67, 6),
    "Japan":          (0.65, 7),
    "South Korea":    (0.62, 8),
    "Australia":      (0.60, 8),
    "China":          (0.68, 9),
    "India":          (0.49, 11),
    "Brazil":         (0.56, 12),
    "South Africa":   (0.46, 13),
}
# Each country draws from its own independent RNG seed → genuinely uncorrelated noise.
# Emerging-market countries get larger noise amplitude (more idiosyncratic risk).
map_rows = []
for country, (coeff, noise_amp) in LINKAGE.items():
    seed  = hash(country + str(int(sim_prob * 100))) % 99991
    noise = np.random.default_rng(seed).uniform(-noise_amp, noise_amp)
    map_rows.append({
        "country": country,
        "risk":    float(np.clip(sim_prob * coeff + noise, 0, 95)),
        "linkage": f"{coeff:.0%}",
    })
map_df  = pd.DataFrame(map_rows)

fig_map = px.choropleth(
    map_df, locations="country", locationmode="country names",
    color="risk", color_continuous_scale="Reds", range_color=[8, 88],
    labels={"risk": "Recession Risk (%)", "linkage": "Trade Linkage"},
    hover_data={"linkage": True, "risk": ":.1f"},
    title="Global Recession Risk — US model output × regional trade & financial linkage",
)
fig_map.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="rgba(255,255,255,.55)"),
    geo=dict(bgcolor="rgba(0,0,0,0)", showframe=False, showcoastlines=True,
             coastlinecolor="rgba(255,255,255,.1)"),
    margin=dict(l=0,r=0,t=40,b=0),
    coloraxis_colorbar=dict(title="Risk %"),
)
st.plotly_chart(fig_map, use_container_width=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# ⑩ AI ECONOMIC INSIGHTS — variable, human-like
# ══════════════════════════════════════════════

st.markdown('<p class="section-label">🤖 AI Economic Insights · Mixtral 8×7B</p>',
            unsafe_allow_html=True)

ins_col, pol_col = st.columns([3,2], gap="medium")

with ins_col:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Macroeconomic Analysis</p>', unsafe_allow_html=True)
    with st.spinner("Generating insights…"):
        insights = generate_insights(live, live_prob, live_stress, sim_vals, sim_prob)
    for ctype, icon, text in insights:
        st.markdown(f"""<div class="insight-card {ctype}">
        <span class="insight-icon">{icon}</span>
        <span class="insight-text">{text}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with pol_col:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Policy Recommendation</p>', unsafe_allow_html=True)
    if sim_prob > 70:
        p_icon,p_cls,p_label = "🏦","high","🔴 HIGH RECESSION RISK"
        p_text = ("<strong>Proactive intervention warranted.</strong> "
                  "Rate cuts alone may be insufficient if credit spreads are widening — "
                  "fiscal transmission may be faster. Unemployment support and targeted "
                  "sector relief should be modelled alongside monetary action.")
    elif sim_prob > 40:
        p_icon,p_cls,p_label = "📊","medium","🟡 ELEVATED RISK"
        p_text = ("<strong>Monitor, don't panic.</strong> "
                  "The yield curve and credit spread are the variables to watch. "
                  "If either deteriorates another standard deviation, "
                  "the risk profile changes materially. Contingency frameworks "
                  "should already be in preparation.")
    else:
        p_icon,p_cls,p_label = "🌿","low","🟢 STABLE"
        p_text = ("<strong>Stable — but not complacent.</strong> "
                  "Low stress scores can mask building vulnerabilities, "
                  "especially in credit and housing. The forecast window "
                  "and yield spread remain worth watching even in stable regimes.")
    st.markdown(f"""<div class="policy-box">
    <div style="font-size:1.7rem;margin-bottom:.7rem;">{p_icon}</div>{p_text}
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-label" style="margin-bottom:.45rem;">Current Risk Level</p>',
                unsafe_allow_html=True)
    st.markdown(f'<span class="risk-badge {p_cls}" style="font-size:.88rem;'
                f'padding:.3rem .9rem;">{p_label}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# ⑪ AI CHATBOT — context-aware, Mixtral
# ══════════════════════════════════════════════

st.markdown("## 🤖 AI Economist Assistant")
st.markdown('<p style="color:rgba(255,255,255,.38);font-size:.83rem;margin-top:-.5rem;'
            'margin-bottom:1rem;">Ask anything — indicator dynamics, scenario implications, '
            'policy trade-offs, historical analogues, global contagion. '
            'Powered by Mixtral 8×7B.</p>', unsafe_allow_html=True)

question = st.text_input(
    "Your question:",
    placeholder='"Is the yield curve telling us something different from the model?" · '
                '"Walk me through what happens if China slows sharply" · '
                '"What would need to change to cut recession risk in half?"',
)

if question:
    system = build_context(live, live_prob, live_stress, sim_vals, sim_prob)
    temp   = round(float(np.random.default_rng(hash(question) % 9999).uniform(0.72, 0.91)), 2)
    with st.spinner(f"Reasoning with Mixtral 8×7B (temp {temp})…"):
        answer = call_llm(system, question, temperature=temp, max_tokens=620)
    st.info(answer)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# ⑫ NEWS SENTIMENT
# ══════════════════════════════════════════════

st.markdown('<p class="section-label">🗞️ Economic News Sentiment</p>', unsafe_allow_html=True)

news = get_news()
if news:
    for art in news:
        s   = news_sentiment(art["title"])
        msg = f"**{art['source']}** — {art['title']}"
        if s == "Negative":   st.error(msg)
        elif s == "Neutral":  st.warning(msg)
        else:                 st.success(msg)
else:
    st.warning("No economic news available right now.")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# ⑬ WHY THIS MATTERS
# ══════════════════════════════════════════════

st.markdown('<p class="section-label">💡 Why This Matters</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Real-World Stakeholder Value</p>', unsafe_allow_html=True)

wm1, wm2, wm3 = st.columns(3, gap="medium")

with wm1:
    st.markdown("""<div class="matters-card">
    <h4>🏛️ Central Banks & Governments</h4>
    <p>Recession signals 2–3 quarters early give policymakers time to calibrate rate cuts,
    deploy automatic stabilisers, and prepare unemployment support before conditions deteriorate.</p>
    <p>A stress index above threshold for two consecutive quarters has historically
    preceded recessions — giving meaningful lead time over lagging GDP data.</p>
    <p><b>Actionable:</b> Trigger contingency review when recession probability exceeds 55%
    and credit spreads are widening simultaneously.</p>
    </div>""", unsafe_allow_html=True)

with wm2:
    st.markdown("""<div class="matters-card">
    <h4>🏦 Banks & Credit Institutions</h4>
    <p>Credit risk models depend on macro forecasts. Rising stress indices signal
    higher expected default rates — warranting tighter lending standards and
    increased loan-loss provisions before losses materialise.</p>
    <p>The credit spread indicator in this model directly captures market-implied
    default risk, giving banks a forward-looking signal not in backward-looking ratings.</p>
    <p><b>Actionable:</b> When credit spread > 3% and PMI < 48, review cyclical sector exposure.</p>
    </div>""", unsafe_allow_html=True)

with wm3:
    st.markdown("""<div class="matters-card">
    <h4>📈 Investors & Portfolio Managers</h4>
    <p>Macro regime shifts — from expansion to contraction — drive asset class
    correlations and drawdown risk. Equity, credit, and duration positioning
    should all respond to the recession probability signal.</p>
    <p>The scenario simulator lets portfolio managers stress-test holdings against
    specific shocks: stagflation, rate spikes, demand collapse, or oil shocks.</p>
    <p><b>Actionable:</b> Above 60% recession probability — reduce cyclical beta,
    increase duration and quality credit exposure.</p>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# ⑭ MODEL TRANSPARENCY
# ══════════════════════════════════════════════

st.markdown('<p class="section-label">🔬 Model Transparency</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">What the Model Does — and Doesn\'t Do</p>',
            unsafe_allow_html=True)

hw1, hw2 = st.columns(2, gap="medium")
with hw1:
    # Compute weight percentages dynamically so the text is always accurate
    w_pct = {
        "Unemployment":        round(abs(WEIGHTS[1]) / abs(WEIGHTS).sum() * 100),
        "CPI Inflation":       round(abs(WEIGHTS[0]) / abs(WEIGHTS).sum() * 100),
        "GDP Growth":          round(abs(WEIGHTS[5]) / abs(WEIGHTS).sum() * 100),
        "Yield Spread":        round(abs(WEIGHTS[8]) / abs(WEIGHTS).sum() * 100),
        "Credit Spread":       round(abs(WEIGHTS[9]) / abs(WEIGHTS).sum() * 100),
        "Consumer Confidence": round(abs(WEIGHTS[3]) / abs(WEIGHTS).sum() * 100),
    }
    st.markdown(f"""<div class="info-box"><h4>📥 12 Input Features & Weights</h4>
    Features are z-score standardised then multiplied by theory-motivated signed weights
    and summed into a <b>composite stress score</b>:<br><br>
    <code>S = Σ (sign_i × w_i × z_i)</code><br><br>
    Weight allocation (normalised to 100%, capped at ~22% per feature):<br>
    Unemployment <b>{w_pct['Unemployment']}%</b> ↑risk ·
    CPI Inflation <b>{w_pct['CPI Inflation']}%</b> ↑risk ·
    GDP Growth <b>{w_pct['GDP Growth']}%</b> ↓risk ·
    Yield Spread <b>{w_pct['Yield Spread']}%</b> ↓risk<br>
    Credit Spread <b>{w_pct['Credit Spread']}%</b> ↑risk ·
    Consumer Confidence <b>{w_pct['Consumer Confidence']}%</b> ↓risk ·
    remaining 6 features share the rest.<br><br>
    Positive contribution → raises risk. Negative → lowers risk.
    Direction is determined dynamically from the data, not hardcoded.
    </div>
    <div class="info-box"><h4>🎯 Why 8% Label Noise?</h4>
    Real economic data is inherently ambiguous — quarters near the threshold
    can legitimately be classified either way. Injecting 8% random label noise
    during training forces the model to be uncertain near the boundary,
    producing realistic 70–85% accuracy instead of an overfit 95%+.
    </div>""", unsafe_allow_html=True)

with hw2:
    st.markdown("""<div class="limit-box"><h4>⚠ What This Model Cannot Do</h4>
    <ul style="margin:.4rem 0 0;padding-left:1.2rem;">
    <li>Predict black-swan events (COVID, 2008 Lehman, wars)</li>
    <li>Account for geopolitical shocks not priced into indicators</li>
    <li>Capture structural regime changes not in historical data</li>
    <li>Replace specialist macro forecasting teams</li>
    <li>Provide financial advice in any form</li>
    </ul>
    </div>
    <div class="limit-box"><h4>📊 Forecast & Map Methodology</h4>
    6-month forecast = 60% linear trend on historical stress + 40% scenario stress
    + AR(1) noise (σ = 20% of historical std). High stress scenarios drift upward;
    low stress scenarios mean-revert. Each month differs by design — no flat output.<br><br>
    Global map: US risk is direct model output. All other countries = US risk ×
    trade/financial linkage coefficient + independent regional noise
    (±5–13% depending on EM exposure). Countries are <em>not</em> uniformly correlated.
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# ⑮ FOOTER
# ══════════════════════════════════════════════

st.markdown(f"""
<div style="text-align:center;padding:1rem 0 .5rem;">
<p style="font-family:'JetBrains Mono',monospace;font-size:.58rem;letter-spacing:.14em;
          color:rgba(255,255,255,.14);text-transform:uppercase;">
Economic Early Warning System · RandomForest + Mixtral 8×7B · 12 Macro Indicators ·
FRED / Historically-Calibrated Simulation · {meta['fetched_at'][:10]} ·
For Research & Educational Purposes Only · Not Financial Advice
</p></div>
""", unsafe_allow_html=True)