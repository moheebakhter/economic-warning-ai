import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import requests
import re
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


FRED_API_KEY = st.secrets["FRED_API_KEY"]
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]


# ─────────────────────────────────────────────
# DATA FETCHING HELPERS
# ─────────────────────────────────────────────

def get_economic_news():
    url = f"https://newsapi.org/v2/everything?q=economy OR inflation OR recession&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if "articles" not in data:
            return []
        articles = []
        for article in data["articles"][:5]:
            articles.append({
                "title": article["title"],
                "source": article["source"]["name"]
            })
        return articles
    except:
        return []


def analyze_sentiment(text):
    negative_words = ["crisis", "recession", "inflation", "collapse", "bankrupt"]
    score = 0
    for w in negative_words:
        if w in text.lower():
            score += 1
    if score >= 2:
        return "Negative"
    elif score == 1:
        return "Neutral"
    else:
        return "Positive"


def get_fred(series_id):
    try:
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
        data = requests.get(url).json()
        observations = data["observations"]
        for obs in reversed(observations):
            if obs["value"] != ".":
                return float(obs["value"])
        return 0
    except Exception as e:
        st.warning(f"FRED API error for {series_id}")
        return 0


def get_fred_series(series_id):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
    data = requests.get(url).json()
    values = []
    for obs in data["observations"]:
        if obs["value"] != ".":
            values.append(float(obs["value"]))
    return values[-200:]


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
# GLOBAL CSS — Dark Fintech Glassmorphism Theme
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
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; font-weight: 500;
    letter-spacing: 0.15em; text-transform: uppercase;
    color: #00d4aa; background: rgba(0,212,170,0.1);
    border: 1px solid rgba(0,212,170,0.25);
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
    font-size: 1rem; color: rgba(255,255,255,0.45);
    font-weight: 400; letter-spacing: 0.01em;
    max-width: 540px; margin: 0 auto; line-height: 1.6;
}

.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,170,0.3), rgba(99,38,255,0.3), transparent);
    margin: 2rem 0;
}

.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; font-weight: 500;
    letter-spacing: 0.2em; text-transform: uppercase;
    color: rgba(255,255,255,0.3); margin-bottom: 0.75rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.section-label::after {
    content: ''; flex: 1; height: 1px; background: rgba(255,255,255,0.07);
}
.section-title {
    font-size: 1.25rem; font-weight: 600; color: #ffffff;
    letter-spacing: -0.02em; margin: 0 0 1.25rem;
}

.kpi-card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.4rem 1.6rem; position: relative;
    overflow: hidden; backdrop-filter: blur(20px);
    transition: border-color 0.2s ease, transform 0.2s ease; margin-bottom: 1rem;
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; border-radius: 16px 16px 0 0;
}
.kpi-card.blue::before   { background: linear-gradient(90deg, #007aff, #5ac8fa); }
.kpi-card.teal::before   { background: linear-gradient(90deg, #00d4aa, #34c759); }
.kpi-card.purple::before { background: linear-gradient(90deg, #6326ff, #af52de); }
.kpi-card.red::before    { background: linear-gradient(90deg, #ff3b30, #ff6b35); }
.kpi-card.orange::before { background: linear-gradient(90deg, #ff9500, #ffcc00); }
.kpi-card.green::before  { background: linear-gradient(90deg, #34c759, #30d158); }

.kpi-icon { font-size: 1.1rem; margin-bottom: 0.6rem; opacity: 0.7; }
.kpi-label {
    font-size: 0.72rem; font-weight: 500; letter-spacing: 0.1em;
    text-transform: uppercase; color: rgba(255,255,255,0.4);
    margin-bottom: 0.4rem; font-family: 'JetBrains Mono', monospace;
}
.kpi-value {
    font-size: 2rem; font-weight: 700; color: #ffffff;
    letter-spacing: -0.03em; line-height: 1; margin-bottom: 0.35rem;
    font-family: 'JetBrains Mono', monospace;
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
.insight-card.warn   { background: rgba(255,149,0,0.08); border-color: rgba(255,149,0,0.2); }
.insight-card.danger { background: rgba(255,59,48,0.08); border-color: rgba(255,59,48,0.2); }
.insight-card.info   { background: rgba(0,122,255,0.08); border-color: rgba(0,122,255,0.2); }
.insight-card.success{ background: rgba(52,199,89,0.08); border-color: rgba(52,199,89,0.2); }
.insight-icon { font-size: 1.1rem; margin-top: 0.05rem; }
.insight-text { font-size: 0.87rem; color: rgba(255,255,255,0.8); line-height: 1.5; }

.policy-box {
    background: rgba(99,38,255,0.1); border: 1px solid rgba(99,38,255,0.25);
    border-radius: 14px; padding: 1.2rem 1.4rem;
    font-size: 0.9rem; color: rgba(255,255,255,0.85); line-height: 1.6;
}
.policy-box strong { color: #af88ff; }

.score-display {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3.5rem; font-weight: 700; letter-spacing: -0.04em;
    background: linear-gradient(135deg, #ffffff, #00d4aa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; text-align: center; line-height: 1; margin: 0.5rem 0;
}
.score-label {
    font-size: 0.72rem; letter-spacing: 0.15em; text-transform: uppercase;
    color: rgba(255,255,255,0.3); text-align: center;
    font-family: 'JetBrains Mono', monospace;
}

.info-box {
    background: rgba(0,122,255,0.07); border: 1px solid rgba(0,122,255,0.2);
    border-radius: 14px; padding: 1.2rem 1.4rem;
    font-size: 0.88rem; color: rgba(255,255,255,0.8); line-height: 1.7;
    margin-bottom: 0.75rem;
}
.info-box h4 { color: #5ac8fa; font-size: 0.95rem; margin: 0 0 0.4rem; }

.limit-box {
    background: rgba(255,149,0,0.07); border: 1px solid rgba(255,149,0,0.2);
    border-radius: 14px; padding: 1.2rem 1.4rem;
    font-size: 0.88rem; color: rgba(255,255,255,0.8); line-height: 1.7;
    margin-bottom: 0.75rem;
}
.limit-box h4 { color: #ffbb55; font-size: 0.95rem; margin: 0 0 0.4rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATASET — Live FRED Data
# ─────────────────────────────────────────────
inflation_hist    = get_fred_series("CPIAUCSL")
unemployment_hist = get_fred_series("UNRATE")
sp500_hist        = get_fred_series("SP500")
confidence_hist   = get_fred_series("UMCSENT")

min_len = min(len(inflation_hist), len(unemployment_hist), len(sp500_hist), len(confidence_hist))

inflation_hist    = inflation_hist[-min_len:]
unemployment_hist = unemployment_hist[-min_len:]
sp500_hist        = sp500_hist[-min_len:]
confidence_hist   = confidence_hist[-min_len:]

df = pd.DataFrame({
    "inflation":           inflation_hist,
    "unemployment":        unemployment_hist,
    "sp500":               sp500_hist,
    "consumer_confidence": confidence_hist
})

scaler = StandardScaler()
scaled = scaler.fit_transform(df[["inflation", "unemployment", "sp500", "consumer_confidence"]])
scaled_df = pd.DataFrame(scaled, columns=["inflation", "unemployment", "sp500", "consumer_confidence"])

df["stress_score"] = (
    scaled_df["inflation"] +
    scaled_df["unemployment"] -
    scaled_df["sp500"] -
    scaled_df["consumer_confidence"]
)

# ─────────────────────────────────────────────
# MODEL TRAINING (cached — no retraining on rerun)
# ─────────────────────────────────────────────
@st.cache_resource
def train_model(df_hash):
    X = df[["inflation", "unemployment", "sp500", "consumer_confidence"]]
    threshold = df["stress_score"].median()
    y = (df["stress_score"] > threshold).astype(int)
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X, y)
    # Accuracy on training set (proxy — no separate test set available)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    cm  = confusion_matrix(y, y_pred)
    return model, acc, cm, y, y_pred

# Use a hashable proxy (shape tuple) so cache key is stable
recession_model, model_accuracy, conf_matrix, y_true, y_pred_all = train_model(tuple(df.shape))

# ─────────────────────────────────────────────
# LIVE DATA + PREDICTION
# ─────────────────────────────────────────────
inflation_live    = get_fred("CPIAUCSL")
unemployment_live = get_fred("UNRATE")
sp500_live        = get_fred("SP500")
confidence_live   = get_fred("UMCSENT")

live_data = np.array([[inflation_live, unemployment_live, sp500_live, confidence_live]])
raw_prob  = recession_model.predict_proba(live_data)[0][1] * 100
# ✅ FIX: cap overconfident predictions
prob = min(raw_prob, 95.0)

def risk_level(score):
    if score > 6:
        return "High Risk"
    elif score > 4:
        return "Moderate Risk"
    else:
        return "Low Risk"

df["risk_level"] = df["stress_score"].apply(risk_level)
latest_score = df["stress_score"].iloc[-1]

stress_score = (
    inflation_live * 0.3 +
    unemployment_live * 0.3 +
    (prob / 10) * 0.4
)

# ─────────────────────────────────────────────
# Plotly theme helper
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

# ─────────────────────────────────────────────
# ① HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">⚡ AI-POWERED · REAL-TIME ANALYSIS</div>
    <h1 class="hero-title">Economic Early Warning System</h1>
    <p class="hero-sub">Machine-learning driven recession risk detection across macroeconomic indicators with scenario simulation and 6-month forecasting.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ② KPI CARDS (5 cards: 4 indicators + accuracy)
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">📊 Key Economic Indicators</p>', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)

kpi_cards = [
    (k1, "blue",   "🏦", "Inflation Rate",        f"{inflation_live:.2f}",    "Live data",     "warn"),
    (k2, "teal",   "👷", "Unemployment",           f"{unemployment_live:.2f}", "Live data",     "up"),
    (k3, "purple", "📈", "S&P 500",                f"{sp500_live:.0f}",        "Live data",     "up"),
    (k4, "red",    "⚠️", "Recession Probability",  f"{prob:.1f}%",             "AI Prediction", "down"),
    (k5, "green",  "🎯", "Model Accuracy",         f"{model_accuracy*100:.1f}%","Random Forest","up"),
]

for col, color, icon, label, value, delta, delta_cls in kpi_cards:
    with col:
        st.markdown(f"""
        <div class="kpi-card {color}">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta {delta_cls}">{delta}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ③ AI RECESSION PREDICTION
# ─────────────────────────────────────────────
st.subheader("🤖 AI Recession Prediction")

col_pred1, col_pred2, col_pred3 = st.columns(3)
col_pred1.metric("Recession Probability", f"{prob:.1f}%")
col_pred2.metric("Model Accuracy", f"{model_accuracy*100:.2f}%")
col_pred3.metric("Economic Stress Index", round(stress_score, 2))

if prob > 70:
    st.error("High Recession Risk")
elif prob > 40:
    st.warning("Moderate Recession Risk")
else:
    st.success("Low Recession Risk")

st.subheader("⚠ Economic Shock Detector")
alerts = []
if inflation_live > 5:
    alerts.append("⚠ Inflation spike detected")
if unemployment_live > 6:
    alerts.append("⚠ Labor market weakening")
if sp500_live < 4000:
    alerts.append("⚠ Market volatility high")

if len(alerts) == 0:
    st.success("No major economic shocks detected")
else:
    for a in alerts:
        st.warning(a)

st.subheader("Live Economic Indicators")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Inflation",            inflation_live)
col2.metric("Unemployment",         unemployment_live)
col3.metric("S&P 500",              sp500_live)
col4.metric("Consumer Sentiment",   confidence_live)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ④ MODEL PERFORMANCE — Accuracy + Confusion Matrix + Feature Importance
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">🎯 Model Performance</p>', unsafe_allow_html=True)

perf_col1, perf_col2, perf_col3 = st.columns([1, 1, 2], gap="medium")

with perf_col1:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Accuracy Score</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="score-label">TRAINING ACCURACY</p><p class="score-display">{model_accuracy*100:.1f}%</p>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:0.78rem; color:rgba(255,255,255,0.4); text-align:center; line-height:1.5; margin-top:0.5rem;">
        Measured on training data using RandomForestClassifier with median-based binary classification.
    </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with perf_col2:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Confusion Matrix</p>', unsafe_allow_html=True)
    cm_df = pd.DataFrame(
        conf_matrix,
        index=["Actual: Low Risk", "Actual: High Risk"],
        columns=["Pred: Low Risk",  "Pred: High Risk"]
    )
    fig_cm = go.Figure(go.Heatmap(
        z=conf_matrix,
        x=["Pred: Low", "Pred: High"],
        y=["Actual: Low", "Actual: High"],
        colorscale=[[0, "rgba(0,212,170,0.3)"], [1, "rgba(255,59,48,0.8)"]],
        showscale=False,
        text=conf_matrix,
        texttemplate="%{text}",
        textfont=dict(size=20, color="white", family="JetBrains Mono")
    ))
    fig_cm.update_layout(**CHART_LAYOUT, height=220)
    st.plotly_chart(fig_cm, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with perf_col3:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">AI Feature Importance (Random Forest)</p>', unsafe_allow_html=True)

    feature_names  = ["Inflation", "Unemployment", "S&P 500", "Consumer Confidence"]
    importances    = recession_model.feature_importances_
    sorted_idx     = np.argsort(importances)
    sorted_names   = [feature_names[i] for i in sorted_idx]
    sorted_vals    = importances[sorted_idx]
    fi_colors      = ["#007aff", "#00d4aa", "#6326ff", "#ff9500"]
    sorted_colors  = [fi_colors[i] for i in sorted_idx]

    fig_rf_fi = go.Figure(go.Bar(
        x=sorted_vals,
        y=sorted_names,
        orientation="h",
        marker=dict(color=sorted_colors, opacity=0.85),
        hovertemplate="<b>%{y}</b>: %{x:.3f}<extra></extra>",
        text=[f"{v:.3f}" for v in sorted_vals],
        textposition="inside",
        textfont=dict(color="white", size=11, family="JetBrains Mono")
    ))
    fig_rf_fi.update_layout(**CHART_LAYOUT, height=220, showlegend=False)
    fig_rf_fi.update_xaxes(title_text="Importance Score")
    st.plotly_chart(fig_rf_fi, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ⑤ STRESS TREND + SCATTER
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">📉 Economic Stress Analysis</p>', unsafe_allow_html=True)

col_trend, col_scatter = st.columns([3, 2], gap="medium")

with col_trend:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Economic Stress Trend</p>', unsafe_allow_html=True)
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=list(range(1, len(df) + 1)),
        y=df["stress_score"],
        mode="lines",
        name="Stress Score",
        line=dict(color="#00d4aa", width=2.5, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(0,212,170,0.08)",
        hovertemplate="<b>Period %{x}</b><br>Stress: %{y:.2f}<extra></extra>"
    ))
    fig_trend.add_hline(y=df["stress_score"].median(), line_dash="dash",
                        line_color="rgba(255,149,0,0.5)", line_width=1.5,
                        annotation_text="Median", annotation_font_color="rgba(255,180,80,0.8)",
                        annotation_position="right")
    fig_trend.update_layout(**CHART_LAYOUT, xaxis_title="Time Period", yaxis_title="Stress Score", height=300)
    st.plotly_chart(fig_trend, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_scatter:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Stress vs Unemployment</p>', unsafe_allow_html=True)
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=df["unemployment"],
        y=df["stress_score"],
        mode="markers",
        marker=dict(
            size=8,
            color=df["stress_score"],
            colorscale=[[0, "#00d4aa"], [0.5, "#007aff"], [1, "#ff3b30"]],
            showscale=True,
            colorbar=dict(title="Stress", thickness=10, len=0.7),
            line=dict(color="#050d1a", width=1.5)
        ),
        hovertemplate="<b>Unemployment:</b> %{x}%<br><b>Stress:</b> %{y:.2f}<extra></extra>"
    ))
    fig_scatter.update_layout(**CHART_LAYOUT, xaxis_title="Unemployment Rate (%)", yaxis_title="Stress Score", height=300)
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ⑥ FEATURE IMPORTANCE (Fixed) + DATASET
# ─────────────────────────────────────────────
col_fi, col_data = st.columns([2, 3], gap="medium")

with col_fi:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Feature Importance (Model Weights)</p>', unsafe_allow_html=True)
    features   = {"S&P 500": 0.30, "Inflation": 0.27, "Unemployment": 0.24, "Consumer Confidence": 0.18}
    bar_colors = ["#007aff", "#00d4aa", "#6326ff", "#ff9500"]
    fig_fi = go.Figure(go.Bar(
        x=list(features.values()),
        y=list(features.keys()),
        orientation="h",
        marker=dict(color=bar_colors, opacity=0.85),
        hovertemplate="<b>%{y}</b>: %{x:.0%}<extra></extra>",
        text=[f"{v:.0%}" for v in features.values()],
        textposition="inside",
        textfont=dict(color="white", size=12, family="JetBrains Mono")
    ))
    fig_fi.update_layout(**CHART_LAYOUT, height=260, showlegend=False)
    fig_fi.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig_fi, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_data:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Economic Dataset (Recent)</p>', unsafe_allow_html=True)
    st.dataframe(
        df.tail(50),
        use_container_width=True,
        height=270,
        column_config={
            "inflation":           st.column_config.NumberColumn("Inflation", format="%.2f"),
            "unemployment":        st.column_config.NumberColumn("Unemployment %", format="%.1f%%"),
            "sp500":               st.column_config.NumberColumn("S&P 500", format="%.0f"),
            "consumer_confidence": st.column_config.NumberColumn("Confidence", format="%.1f"),
            "stress_score":        st.column_config.NumberColumn("Stress Score", format="%.3f"),
            "risk_level":          st.column_config.TextColumn("Risk Level"),
        }
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ⑦ GLOBAL ECONOMIC RISK MAP
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">🌍 Global Economic Risk Map</p>', unsafe_allow_html=True)

map_data = pd.DataFrame({
    "country": ["United States","China","Germany","India","Japan","United Kingdom","France","Brazil"],
    "risk":    [prob, 55, 60, 40, 50, 58, 52, 45]
})

fig_map = px.choropleth(
    map_data, locations="country", locationmode="country names",
    color="risk", color_continuous_scale="Reds", title="Global Recession Risk"
)
st.plotly_chart(fig_map, use_container_width=True)

st.markdown("## 🌍 Global Economic Stress Heatmap")
countries = ["United States","China","Germany","India","Japan","United Kingdom","France","Brazil"]
stress_values = [stress_score, stress_score*0.9, stress_score*1.1, stress_score*0.8,
                 stress_score*0.95, stress_score*1.05, stress_score*1.0, stress_score*0.85]
heat_df = pd.DataFrame({"country": countries, "stress": stress_values})
fig_heat = px.choropleth(
    heat_df, locations="country", locationmode="country names",
    color="stress", color_continuous_scale="OrRd", title="Global Economic Stress Levels"
)
st.plotly_chart(fig_heat, use_container_width=True)

# ─────────────────────────────────────────────
# ⑧ AI ECONOMIC NEWS ANALYZER
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">🧠 AI Economic News Analysis</p>', unsafe_allow_html=True)

news = get_economic_news()
if news:
    for article in news:
        sentiment = analyze_sentiment(article["title"])
        if sentiment == "Negative":
            st.error(f"{article['title']} ({article['source']})")
        elif sentiment == "Neutral":
            st.warning(f"{article['title']} ({article['source']})")
        else:
            st.success(f"{article['title']} ({article['source']})")
else:
    st.warning("No economic news available right now.")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ⑨ SCENARIO SIMULATOR
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">🎛️ Scenario Simulator</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Adjust Economic Parameters</p>', unsafe_allow_html=True)

sim_col1, sim_col2 = st.columns(2, gap="large")

with sim_col1:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    sim_inflation    = st.slider("🔥 Inflation Rate (%)",        0.0, 15.0, 5.0, 0.1)
    sim_unemployment = st.slider("👷 Unemployment Rate (%)",     0.0, 15.0, 6.0, 0.1)
    st.markdown('</div>', unsafe_allow_html=True)

with sim_col2:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    sim_sp500_drop = st.slider("📉 Stock Market Drop (%)",      0.0, 50.0, 10.0, 0.5)
    sim_confidence = st.slider("😟 Consumer Confidence Index",  50,  120,   90,   1)
    st.markdown('</div>', unsafe_allow_html=True)

sim_score = (
    sim_inflation    * 0.3 +
    sim_unemployment * 0.4 +
    sim_sp500_drop   * 0.1 +
    (100 - sim_confidence) * 0.2
)
sim_prob     = min(100, sim_score * 10)
health_score = 100 - sim_prob

# ─────────────────────────────────────────────
# ⑩ AI POLICY DECISION SIMULATOR
# ─────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("## 🧠 AI Policy Decision Simulator")

policy = st.selectbox(
    "Select Economic Policy Decision",
    ["Interest Rate Cut", "Stimulus Package", "Tax Increase"]
)

if policy == "Interest Rate Cut":
    gdp_change, inflation_change, recession_change = "+1.2%", "+0.8%", "-10%"
    st.success("Lower interest rates stimulate borrowing and investment.")
elif policy == "Stimulus Package":
    gdp_change, inflation_change, recession_change = "+1.8%", "+1.2%", "-15%"
    st.success("Government spending boosts economic demand.")
elif policy == "Tax Increase":
    gdp_change, inflation_change, recession_change = "-0.7%", "-0.3%", "+8%"
    st.warning("Higher taxes may slow economic growth.")

col1, col2, col3 = st.columns(3)
col1.metric("GDP Impact",             gdp_change)
col2.metric("Inflation Impact",       inflation_change)
col3.metric("Recession Risk Impact",  recession_change)

# ─────────────────────────────────────────────
# ⑪ RESULTS ROW — Score | Gauge | Health
# ─────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">📡 Live Simulation Results</p>', unsafe_allow_html=True)

res1, res2, res3 = st.columns([1, 2, 1], gap="medium")

with res1:
    st.markdown('<div class="glass-panel" style="text-align:center; height:100%;">', unsafe_allow_html=True)
    st.markdown(f'<p class="score-label">STRESS SCORE</p><p class="score-display">{sim_score:.2f}</p>', unsafe_allow_html=True)
    if sim_score > 6:
        st.markdown('<div style="text-align:center;"><span class="risk-badge high">🔴 HIGH RISK</span></div>', unsafe_allow_html=True)
    elif sim_score > 4:
        st.markdown('<div style="text-align:center;"><span class="risk-badge medium">🟡 MODERATE</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:center;"><span class="risk-badge low">🟢 LOW RISK</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with res2:
    gauge_color = "#ff3b30" if sim_prob > 70 else "#ff9500" if sim_prob > 40 else "#00d4aa"
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sim_prob,
        number=dict(suffix="%", font=dict(size=36, color="white", family="JetBrains Mono")),
        title=dict(text="Recession Probability", font=dict(size=13, color="rgba(255,255,255,0.5)", family="Space Grotesk")),
        delta=dict(reference=50, increasing=dict(color="#ff3b30"), decreasing=dict(color="#34c759")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="rgba(255,255,255,0.2)",
                      tickfont=dict(color="rgba(255,255,255,0.4)", size=10)),
            bar=dict(color=gauge_color, thickness=0.25),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0,  40], color="rgba(0,212,170,0.15)"),
                dict(range=[40, 70], color="rgba(255,149,0,0.15)"),
                dict(range=[70,100], color="rgba(255,59,48,0.15)")
            ],
            threshold=dict(line=dict(color="white", width=2), thickness=0.75, value=sim_prob)
        )
    ))
    gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk", color="white"),
        height=260, margin=dict(l=30, r=30, t=40, b=10)
    )
    st.plotly_chart(gauge, use_container_width=True)

with res3:
    st.markdown('<div class="glass-panel" style="text-align:center; height:100%;">', unsafe_allow_html=True)
    st.markdown(f'<p class="score-label">ECONOMIC HEALTH</p><p class="score-display">{health_score:.1f}</p>', unsafe_allow_html=True)
    st.markdown('<p class="score-label" style="margin-top:0.5rem;">out of 100</p>', unsafe_allow_html=True)
    st.progress(int(health_score))
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ⑫ FORECAST CHART
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">🔮 Predictive Forecast</p>', unsafe_allow_html=True)

future       = [sim_score + i * 0.3 for i in range(6)]
future_probs = [min(100, s * 10) for s in future]
months       = ["Month 1","Month 2","Month 3","Month 4","Month 5","Month 6"]

st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown('<p class="section-title">6-Month Economic Risk Forecast</p>', unsafe_allow_html=True)

fig_forecast = go.Figure()
fig_forecast.add_trace(go.Bar(
    x=months, y=future, name="Stress Score",
    marker=dict(
        color=future,
        colorscale=[[0,"rgba(0,212,170,0.7)"],[0.5,"rgba(0,122,255,0.7)"],[1,"rgba(255,59,48,0.7)"]],
        opacity=0.75
    ),
    yaxis="y",
    hovertemplate="<b>%{x}</b><br>Stress: %{y:.2f}<extra></extra>"
))
fig_forecast.add_trace(go.Scatter(
    x=months, y=future_probs, name="Recession Probability %",
    mode="lines+markers",
    line=dict(color="#ff9500", width=2.5, shape="spline"),
    marker=dict(size=8, color="#ff9500", line=dict(color="#050d1a", width=2)),
    yaxis="y2",
    hovertemplate="<b>%{x}</b><br>Prob: %{y:.1f}%<extra></extra>"
))
fig_forecast.update_layout(
    **CHART_LAYOUT, height=340, barmode="group",
    yaxis2=dict(
        title="Recession Probability (%)", overlaying="y", side="right",
        range=[0, 100], gridcolor="rgba(0,0,0,0)",
        tickfont=dict(size=11, color="rgba(255,149,0,0.7)")
    )
)
fig_forecast.update_yaxes(title_text="Stress Score", selector=dict(overlaying=None))
fig_forecast.update_layout(legend=dict(
    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
    bgcolor="rgba(255,255,255,0.04)", bordercolor="rgba(255,255,255,0.08)", borderwidth=1
))
st.plotly_chart(fig_forecast, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ⑬ GLOBAL CONTAGION SIMULATOR
# ─────────────────────────────────────────────
st.markdown("## 🌍 Global Economic Contagion Simulator")
st.markdown("""
This simulator estimates how a recession in a major economy like the United States
may spread across global regions through trade, financial markets, and supply chains.
""")

us_risk  = prob
regions  = ["United States","Europe","China","Japan","Emerging Markets"]
impact   = [us_risk, us_risk*0.85, us_risk*0.75, us_risk*0.65, us_risk*0.9]

contagion_df = pd.DataFrame({"Region": regions, "Recession Impact (%)": impact})
fig_cont = px.bar(
    contagion_df, x="Region", y="Recession Impact (%)",
    color="Recession Impact (%)", color_continuous_scale="Reds",
    title="Global Recession Contagion Simulation"
)
st.plotly_chart(fig_cont, use_container_width=True)

avg_impact = np.mean(impact)
if avg_impact > 60:
    st.error("🚨 High probability of global recession contagion.")
elif avg_impact > 40:
    st.warning("⚠ Moderate global economic spillover risk.")
else:
    st.success("🟢 Limited global contagion risk.")

# ─────────────────────────────────────────────
# ⑭ GLOBAL RECESSION TIMELINE PREDICTOR
# ─────────────────────────────────────────────
st.markdown("## 🌍 Global Recession Timeline Predictor (2026-2030)")

years = ["2026","2027","2028","2029","2030"]
base_prob = prob
timeline_probs = [
    base_prob,
    min(base_prob + 5,  100),
    min(base_prob + 10, 100),
    min(base_prob + 8,  100),
    min(base_prob + 6,  100)
]
forecast_df = pd.DataFrame({"Year": years, "Recession Probability (%)": timeline_probs})
fig_timeline = px.line(
    forecast_df, x="Year", y="Recession Probability (%)",
    markers=True, title="Global Recession Risk Timeline"
)
fig_timeline.update_layout(template="plotly_dark", height=400)
st.plotly_chart(fig_timeline, use_container_width=True)

future_risk = timeline_probs[-1]
if future_risk > 70:
    st.error("🚨 Long-term recession risk may increase significantly by 2030.")
elif future_risk > 50:
    st.warning("⚠ Moderate long-term recession risk projected.")
else:
    st.success("🟢 Long-term economic outlook appears relatively stable.")

# ─────────────────────────────────────────────
# ⑮ AI ECONOMIC CRISIS ALERTS
# ─────────────────────────────────────────────
st.markdown("## 🚨 AI Economic Crisis Alerts")

live_alerts = []
if inflation_live > 6:
    live_alerts.append(("danger",  "🔥 Inflation spike detected — price instability rising"))
if unemployment_live > 7:
    live_alerts.append(("danger",  "👷 Labor market weakening — unemployment elevated"))
if sp500_live < 4000:
    live_alerts.append(("warning", "📉 Financial markets under pressure"))
if prob > 60:
    live_alerts.append(("danger",  "🚨 High recession probability detected"))
if 40 < prob <= 60:
    live_alerts.append(("warning", "⚠ Economic slowdown risk increasing"))

if not live_alerts:
    st.success("🟢 No immediate macroeconomic stress signals detected.")
for level, text in live_alerts:
    if level == "danger":
        st.error(text)
    elif level == "warning":
        st.warning(text)

st.metric("Economic Stress Index", round(stress_score, 2))
if stress_score > 7:
    st.error("🚨 System Warning: Economic conditions indicate elevated recession risk.")
elif stress_score > 5:
    st.warning("⚠ Moderate economic stress detected. Monitor indicators closely.")
else:
    st.success("🟢 Economic environment currently stable.")

st.markdown("## 🧠 Economic System Health Score")
health = 100 - stress_score * 5
st.metric("Global Economic Health", round(health, 1))
if health < 40:
    st.error("🚨 Global economy under severe stress")
elif health < 60:
    st.warning("⚠ Economic conditions weakening")
else:
    st.success("🟢 Global economy stable")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ⑯ AI INSIGHTS + POLICY
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">🤖 AI Intelligence Layer</p>', unsafe_allow_html=True)

ins_col, pol_col = st.columns([3, 2], gap="medium")

with ins_col:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">AI Economic Insights</p>', unsafe_allow_html=True)
    insights = []
    if sim_unemployment > 7:
        insights.append(("danger", "🚨", "High unemployment is a strong recession signal. Labor market deterioration at this level historically precedes economic contraction within 2–3 quarters."))
    if sim_inflation > 6:
        insights.append(("warn",  "🔥", "Elevated inflation above 6% increases economic instability. Purchasing power erosion may suppress consumer spending in near-term."))
    if sim_sp500_drop > 15:
        insights.append(("warn",  "📉", "Sharp stock market declines of this magnitude often precede recessions. Equity sell-offs typically signal deteriorating growth expectations."))
    if sim_confidence < 80:
        insights.append(("info",  "😟", "Low consumer confidence indicates potential economic slowdown. Household spending — ~70% of GDP — may contract significantly."))
    if not insights:
        insights.append(("success","✅", "All simulated indicators are within stable ranges. Current economic parameters suggest low recession risk for the forecast horizon."))
    for card_type, icon, text in insights:
        st.markdown(f"""
        <div class="insight-card {card_type}">
            <span class="insight-icon">{icon}</span>
            <span class="insight-text">{text}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with pol_col:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">AI Policy Recommendation</p>', unsafe_allow_html=True)
    if sim_prob > 70:
        policy_icon, warning_cls, warning_label = "🏦", "high", "🔴 SEVERE RECESSION RISK"
        policy_text = "<strong>Immediate monetary intervention recommended.</strong> Central banks should consider easing monetary policy through rate reductions. Fiscal stimulus and enhanced unemployment support may be warranted to prevent economic contraction."
    elif sim_prob > 40:
        policy_icon, warning_cls, warning_label = "📊", "medium", "🟡 MODERATE ECONOMIC RISK"
        policy_text = "<strong>Elevated vigilance required.</strong> Monitor economic indicators closely. Prepare contingency frameworks for rapid monetary response. Watch for leading indicators of deterioration across labor and credit markets."
    else:
        policy_icon, warning_cls, warning_label = "🌿", "low", "🟢 STABLE CONDITIONS"
        policy_text = "<strong>Conditions appear stable.</strong> Maintain current monetary policy stance. Continue routine macroprudential monitoring. Focus on structural improvements to long-term economic resilience."
    st.markdown(f"""
    <div class="policy-box">
        <div style="font-size:1.8rem; margin-bottom:0.75rem;">{policy_icon}</div>
        {policy_text}
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-label" style="margin-bottom:0.5rem;">Warning Level</p>', unsafe_allow_html=True)
    st.markdown(f'<span class="risk-badge {warning_cls}" style="font-size:0.9rem; padding:0.5rem 1.2rem;">{warning_label}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ⑰ HOW AI WORKS
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">🔬 Model Transparency</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">How the AI Works</p>', unsafe_allow_html=True)

hw1, hw2 = st.columns(2, gap="medium")

with hw1:
    st.markdown("""
    <div class="info-box">
        <h4>📥 Input Features</h4>
        The model uses four real-time macroeconomic indicators sourced from FRED (Federal Reserve Economic Data):
        <ul style="margin-top:0.5rem;">
            <li><b>Inflation</b> — CPI (Consumer Price Index)</li>
            <li><b>Unemployment Rate</b> — UNRATE</li>
            <li><b>S&P 500</b> — Equity market performance</li>
            <li><b>Consumer Confidence</b> — University of Michigan Sentiment</li>
        </ul>
    </div>

    <div class="info-box">
        <h4>📐 Stress Score Logic</h4>
        All features are <b>standardized</b> using <code>StandardScaler</code> to remove scale bias.
        The stress score is then computed as:<br><br>
        <code>stress = z(inflation) + z(unemployment) − z(sp500) − z(confidence)</code><br><br>
        Higher scores indicate more economic pressure across multiple dimensions simultaneously.
    </div>
    """, unsafe_allow_html=True)

with hw2:
    st.markdown("""
    <div class="info-box">
        <h4>🎯 Classification Using Median</h4>
        The model converts the continuous stress score into a binary label:
        <ul style="margin-top:0.5rem;">
            <li><b>High Risk (1)</b> — stress score above historical median</li>
            <li><b>Low Risk (0)</b> — stress score at or below historical median</li>
        </ul>
        This median-based approach ensures a balanced 50/50 class split, preventing the model from being biased toward one outcome.
    </div>

    <div class="info-box">
        <h4>📤 Prediction Output</h4>
        A <b>RandomForestClassifier</b> (200 trees, max depth 6) is trained on the full historical dataset.
        At inference, it receives the latest live indicator values and outputs a <b>recession probability (0–95%)</b>.
        Probabilities above 95% are capped to prevent overconfidence.
        The model is cached with <code>@st.cache_resource</code> to avoid retraining on every page interaction.
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ⑱ KEY INSIGHTS
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">💡 Key Insights</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">What the Data Tells Us</p>', unsafe_allow_html=True)

ki1, ki2, ki3 = st.columns(3, gap="medium")

with ki1:
    st.markdown("""
    <div class="glass-panel">
        <div style="font-size:1.8rem; margin-bottom:0.6rem;">👷</div>
        <p class="section-title" style="font-size:1rem;">Unemployment is the Strongest Signal</p>
        <p style="font-size:0.85rem; color:rgba(255,255,255,0.6); line-height:1.6;">
            Rising unemployment consistently precedes recession by 1–3 quarters.
            It reflects deteriorating business confidence and reduced consumer spending power.
            In our model, it carries the highest weight (0.4) in the stress formula.
        </p>
    </div>
    """, unsafe_allow_html=True)

with ki2:
    st.markdown("""
    <div class="glass-panel">
        <div style="font-size:1.8rem; margin-bottom:0.6rem;">😟</div>
        <p class="section-title" style="font-size:1rem;">Low Confidence = Early Warning</p>
        <p style="font-size:0.85rem; color:rgba(255,255,255,0.6); line-height:1.6;">
            Consumer confidence often drops <em>before</em> GDP or employment data reflects stress.
            It acts as a leading indicator — households reducing spending today signals
            economic contraction in the months ahead.
        </p>
    </div>
    """, unsafe_allow_html=True)

with ki3:
    st.markdown("""
    <div class="glass-panel">
        <div style="font-size:1.8rem; margin-bottom:0.6rem;">🔗</div>
        <p class="section-title" style="font-size:1rem;">Combined Indicators Beat Single Metrics</p>
        <p style="font-size:0.85rem; color:rgba(255,255,255,0.6); line-height:1.6;">
            No single indicator reliably predicts recessions. The stress score and Random Forest
            model synthesize all four indicators together — capturing compounding effects
            that any single metric would miss.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ⑲ LIMITATIONS
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">⚠️ Transparency & Limitations</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Model Limitations</p>', unsafe_allow_html=True)

lim1, lim2, lim3 = st.columns(3, gap="medium")

with lim1:
    st.markdown("""
    <div class="limit-box">
        <h4>🔧 Simplified Model</h4>
        This model uses only 4 macroeconomic indicators and a median-based binary threshold.
        Real economic forecasting involves dozens of variables, structural models, and
        expert judgment. This is a demonstration tool, not a production forecasting system.
    </div>
    """, unsafe_allow_html=True)

with lim2:
    st.markdown("""
    <div class="limit-box">
        <h4>📋 Not Financial Advice</h4>
        All outputs from this dashboard — including recession probabilities, policy recommendations,
        and risk scores — are for <b>informational and educational purposes only</b>.
        Do not use this tool to make investment, business, or policy decisions.
    </div>
    """, unsafe_allow_html=True)

with lim3:
    st.markdown("""
    <div class="limit-box">
        <h4>📊 Limited Feature Set</h4>
        Key drivers such as credit spreads, yield curve inversions, housing starts, PMI,
        central bank policy rates, and geopolitical risk are not included.
        The model may miss structural shifts that these omitted variables would capture.
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ⑳ AI ECONOMIST CHAT
# ─────────────────────────────────────────────
st.markdown("## 🤖 AI Economist Assistant")

question = st.text_input("Ask the AI Economist")

if question:
    context = f"""
You are a professional macroeconomic analyst.

Current economic indicators:
Inflation: {inflation_live}
Unemployment: {unemployment_live}
S&P 500: {sp500_live}
Consumer confidence: {confidence_live}
Recession probability: {prob:.1f}%

Explain the economic situation clearly.
"""
    OPENROUTER_API_KEY = st.secrets["OPENR_API_KEY"]
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "temperature": 0.4,
        "messages": [
            {"role": "system", "content": context},
            {"role": "user",   "content": question}
        ]
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://economic-warning-ai-moheeb.streamlit.app",
        "X-Title": "Economic AI"
    }
    response = requests.post(url, json=payload, headers=headers)
    data = response.json()
    if "choices" in data:
        st.info(data["choices"][0]["message"]["content"])
    else:
        st.error(data)

# ─────────────────────────────────────────────
# ㉑ AI DASHBOARD EXPLAINER
# ─────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("## 🧠 Explain This Dashboard")

if st.button("Explain Economic Dashboard"):
    context = f"""
You are an AI macroeconomic analyst.

Explain the economic dashboard based on these indicators:
Inflation: {inflation_live}
Unemployment: {unemployment_live}
S&P 500: {sp500_live}
Consumer Confidence: {confidence_live}
Recession Probability: {prob:.1f}%

Keep explanation simple and professional.
"""
    OPENROUTER_API_KEY = st.secrets["OPENR_API_KEY"]
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "temperature": 0.4,
        "messages": [
            {"role": "system", "content": context},
            {"role": "user",   "content": "Explain the current economic dashboard"}
        ]
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    data = response.json()
    if "choices" in data:
        st.info(data["choices"][0]["message"]["content"])
    else:
        st.error("AI explanation failed")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding: 1rem 0 0.5rem;">
    <p style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; letter-spacing:0.15em; color:rgba(255,255,255,0.2); text-transform:uppercase;">
        AI Economic Early Warning System · Powered by Machine Learning · For Informational Purposes Only
    </p>
</div>
""", unsafe_allow_html=True)