import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import requests
import re
from sklearn.linear_model import LogisticRegression



FRED_API_KEY = "5ee1a026dfe571b01ad70e63873b2ef8"


NEWS_API_KEY = "a4b161926aac4ca08604a28b26c9291e"


def get_economic_news():

    url = f"https://newsapi.org/v2/everything?q=economy OR inflation OR recession&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"

    data = requests.get(url).json()

    articles = []

    for article in data["articles"][:5]:
        articles.append({
            "title": article["title"],
            "source": article["source"]["name"]
        })

    return articles


def analyze_sentiment(text):

    negative_words = ["crisis","recession","inflation","collapse","bankrupt"]

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

        # last valid value find karo
        for obs in reversed(observations):
            if obs["value"] != ".":
                return float(obs["value"])

        return 0

    except Exception as e:
        st.warning(f"FRED API error for {series_id}")
        return 0

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

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: #050d1a;
    background-image:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(0, 122, 255, 0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 80%, rgba(0, 212, 170, 0.08) 0%, transparent 55%),
        radial-gradient(ellipse 40% 40% at 60% 30%, rgba(99, 38, 255, 0.07) 0%, transparent 50%);
    min-height: 100vh;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 2.5rem 3rem;
    max-width: 1400px;
}

/* ── Hero Header ── */
.hero-header {
    text-align: center;
    padding: 2.5rem 0 1rem;
    margin-bottom: 0.5rem;
}
.hero-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #00d4aa;
    background: rgba(0, 212, 170, 0.1);
    border: 1px solid rgba(0, 212, 170, 0.25);
    padding: 0.3rem 1rem;
    border-radius: 50px;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.1;
    background: linear-gradient(135deg, #ffffff 0%, #a8c8ff 50%, #00d4aa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.6rem;
}
.hero-sub {
    font-size: 1rem;
    color: rgba(255,255,255,0.45);
    font-weight: 400;
    letter-spacing: 0.01em;
    max-width: 540px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Divider ── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,170,0.3), rgba(99,38,255,0.3), transparent);
    margin: 2rem 0;
}

/* ── Section Labels ── */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.3);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.07);
}
.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #ffffff;
    letter-spacing: -0.02em;
    margin: 0 0 1.25rem;
}

/* ── KPI Cards ── */
.kpi-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
    transition: border-color 0.2s ease, transform 0.2s ease;
    margin-bottom: 1rem;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 16px 16px 0 0;
}
.kpi-card.blue::before   { background: linear-gradient(90deg, #007aff, #5ac8fa); }
.kpi-card.teal::before   { background: linear-gradient(90deg, #00d4aa, #34c759); }
.kpi-card.purple::before { background: linear-gradient(90deg, #6326ff, #af52de); }
.kpi-card.red::before    { background: linear-gradient(90deg, #ff3b30, #ff6b35); }
.kpi-card.orange::before { background: linear-gradient(90deg, #ff9500, #ffcc00); }

.kpi-icon {
    font-size: 1.1rem;
    margin-bottom: 0.6rem;
    opacity: 0.7;
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.4);
    margin-bottom: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 0.35rem;
    font-family: 'JetBrains Mono', monospace;
}
.kpi-delta {
    font-size: 0.75rem;
    font-weight: 500;
    color: rgba(255,255,255,0.35);
}
.kpi-delta.up   { color: #34c759; }
.kpi-delta.down { color: #ff3b30; }
.kpi-delta.warn { color: #ff9500; }

/* ── Glass Containers ── */
.glass-panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 1.75rem;
    backdrop-filter: blur(20px);
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.glass-panel::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, transparent 60%);
    pointer-events: none;
}

/* ── Risk Badge ── */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 0.35rem 0.9rem;
    border-radius: 50px;
}
.risk-badge.high   { background: rgba(255,59,48,0.15); color: #ff6b6b; border: 1px solid rgba(255,59,48,0.3); }
.risk-badge.medium { background: rgba(255,149,0,0.15); color: #ffbb55; border: 1px solid rgba(255,149,0,0.3); }
.risk-badge.low    { background: rgba(52,199,89,0.15); color: #34c759; border: 1px solid rgba(52,199,89,0.3); }

/* ── Sliders ── */
.stSlider > div > div > div { background: rgba(0,212,170,0.2) !important; }
.stSlider > div > div > div > div { background: #00d4aa !important; }

/* ── Metric override ── */
[data-testid="metric-container"] {
    background: transparent;
    border: none;
    padding: 0;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #00d4aa, #007aff) !important;
    border-radius: 4px;
}
.stProgress > div > div {
    background: rgba(255,255,255,0.07) !important;
    border-radius: 4px;
}

/* ── DataFrames ── */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Insight cards ── */
.insight-card {
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
    padding: 1rem 1.2rem;
    border-radius: 12px;
    margin-bottom: 0.6rem;
    border: 1px solid rgba(255,255,255,0.06);
}
.insight-card.warn {
    background: rgba(255,149,0,0.08);
    border-color: rgba(255,149,0,0.2);
}
.insight-card.danger {
    background: rgba(255,59,48,0.08);
    border-color: rgba(255,59,48,0.2);
}
.insight-card.info {
    background: rgba(0,122,255,0.08);
    border-color: rgba(0,122,255,0.2);
}
.insight-card.success {
    background: rgba(52,199,89,0.08);
    border-color: rgba(52,199,89,0.2);
}
.insight-icon { font-size: 1.1rem; margin-top: 0.05rem; }
.insight-text {
    font-size: 0.87rem;
    color: rgba(255,255,255,0.8);
    line-height: 1.5;
}

/* ── Policy box ── */
.policy-box {
    background: rgba(99,38,255,0.1);
    border: 1px solid rgba(99,38,255,0.25);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    font-size: 0.9rem;
    color: rgba(255,255,255,0.85);
    line-height: 1.6;
}
.policy-box strong { color: #af88ff; }

/* ── Score display ── */
.score-display {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3.5rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, #ffffff, #00d4aa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    line-height: 1;
    margin: 0.5rem 0;
}
.score-label {
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.3);
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATASET (unchanged)
# ─────────────────────────────────────────────
data = {
    "inflation":           [2.1, 2.3, 3.0, 4.5, 5.2, 6.1, 7.4],
    "unemployment":        [4.1, 4.3, 5.0, 6.1, 7.2, 7.8, 8.2],
    "sp500":               [4200, 4100, 3900, 3700, 3500, 3300, 3100],
    "consumer_confidence": [110, 105, 100, 95, 90, 85, 80],
    "stress_score":        [0.27, 1.43, 2.96, 4.89, 6.58, 8.13, 9.72]
}
df = pd.DataFrame(data)

# AI recession model training

X = df[["inflation","unemployment","sp500","consumer_confidence"]]

y = (df["stress_score"] > 5).astype(int)

model = LogisticRegression()

model.fit(X,y)

inflation_live = get_fred("CPIAUCSL")
unemployment_live = get_fred("UNRATE")
sp500_live = get_fred("SP500")
confidence_live = get_fred("UMCSENT")

live_data = np.array([[inflation_live, unemployment_live, sp500_live, confidence_live]])

prob = model.predict_proba(live_data)[0][1] * 100


def risk_level(score):
    if score > 6:
        return "High Risk"
    elif score > 4:
        return "Moderate Risk"
    else:
        return "Low Risk"

df["risk_level"] = df["stress_score"].apply(risk_level)
latest_score = df["stress_score"].iloc[-1]

# ─────────────────────────────────────────────
# Plotly theme helper
# ─────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="rgba(255,255,255,0.6)", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.08)",
        tickfont=dict(size=11)
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.08)",
        tickfont=dict(size=11)
    ),
    hoverlabel=dict(
        bgcolor="rgba(10,20,40,0.95)",
        bordercolor="rgba(0,212,170,0.4)",
        font=dict(family="JetBrains Mono", size=12, color="white")
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.04)",
        bordercolor="rgba(255,255,255,0.08)",
        borderwidth=1
    )
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
# ② KPI CARDS
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">📊 Key Economic Indicators</p>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)

kpi_cards = [
    (k1, "blue",   "🏦", "Inflation Rate",       f"{inflation_live:.2f}", "Live data", "warn"),
    (k2, "teal",   "👷", "Unemployment",         f"{unemployment_live:.2f}", "Live data", "up"),
    (k3, "purple", "📈", "S&P 500",              f"{sp500_live:.0f}", "Live data", "up"),
    (k4, "red",    "⚠️", "Recession Probability", f"{prob:.1f}%", "AI Prediction", "down"),
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

st.metric("Recession Probability", f"{prob:.1f}%")

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



# LIVE ECONOMIC DATA
st.subheader("Live Economic Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Inflation", inflation_live)
col2.metric("Unemployment", unemployment_live)
col3.metric("S&P 500", sp500_live)
col4.metric("Consumer Sentiment", confidence_live)


# ─────────────────────────────────────────────
# ③ STRESS TREND + SCATTER
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">📉 Economic Stress Analysis</p>', unsafe_allow_html=True)


col_trend, col_scatter = st.columns([3, 2], gap="medium")

with col_trend:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Economic Stress Trend</p>', unsafe_allow_html=True)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=list(range(1, 8)),
        y=df["stress_score"],
        mode="lines+markers",
        name="Stress Score",
        line=dict(color="#00d4aa", width=2.5, shape="spline"),
        marker=dict(size=8, color="#00d4aa", line=dict(color="#050d1a", width=2)),
        fill="tozeroy",
        fillcolor="rgba(0,212,170,0.08)",
        hovertemplate="<b>Period %{x}</b><br>Stress: %{y:.2f}<extra></extra>"
    ))
    fig_trend.add_hline(y=6, line_dash="dash", line_color="rgba(255,59,48,0.5)", line_width=1.5,
                        annotation_text="High Risk", annotation_font_color="rgba(255,100,100,0.8)",
                        annotation_position="right")
    fig_trend.add_hline(y=4, line_dash="dash", line_color="rgba(255,149,0,0.5)", line_width=1.5,
                        annotation_text="Moderate Risk", annotation_font_color="rgba(255,180,80,0.8)",
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
            size=12,
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
# ④ FEATURE IMPORTANCE + DATASET
# ─────────────────────────────────────────────
col_fi, col_data = st.columns([2, 3], gap="medium")

with col_fi:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Feature Importance</p>', unsafe_allow_html=True)

    features = {
        "S&P 500": 0.30,
        "Inflation": 0.27,
        "Unemployment": 0.24,
        "Consumer Confidence": 0.18
    }
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
    # ✅ FIX: use update_xaxes separately to avoid duplicate xaxis key conflict
    fig_fi.update_layout(**CHART_LAYOUT, height=260, showlegend=False)
    fig_fi.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig_fi, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_data:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Economic Dataset</p>', unsafe_allow_html=True)
    st.dataframe(
        df,
        use_container_width=True,
        height=270,
        column_config={
            "inflation":           st.column_config.NumberColumn("Inflation %", format="%.1f%%"),
            "unemployment":        st.column_config.NumberColumn("Unemployment %", format="%.1f%%"),
            "sp500":               st.column_config.NumberColumn("S&P 500", format="%d"),
            "consumer_confidence": st.column_config.NumberColumn("Confidence", format="%d"),
            "stress_score":        st.column_config.ProgressColumn("Stress Score", min_value=0, max_value=10),
            "risk_level":          st.column_config.TextColumn("Risk Level"),
        }
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)



# ⑧ GLOBAL ECONOMIC RISK MAP
# ─────────────────────────────────────────────

st.markdown('<p class="section-label">🌍 Global Economic Risk Map</p>', unsafe_allow_html=True)

map_data = pd.DataFrame({
    "country": [
        "United States",
        "China",
        "Germany",
        "India",
        "Japan",
        "United Kingdom",
        "France",
        "Brazil"
    ],
    "risk": [
        prob,   # AI predicted risk
        55,
        60,
        40,
        50,
        58,
        52,
        45
    ]
})

fig = px.choropleth(
    map_data,
    locations="country",
    locationmode="country names",
    color="risk",
    color_continuous_scale="Reds",
    title="Global Recession Risk"
)

st.plotly_chart(fig, use_container_width=True)


# ⑨ AI ECONOMIC NEWS ANALYZER
# ─────────────────────────────────────────────

st.markdown('<p class="section-label">🧠 AI Economic News Analysis</p>', unsafe_allow_html=True)

news = get_economic_news()

for article in news:

    sentiment = analyze_sentiment(article["title"])

    if sentiment == "Negative":
        st.error(f"{article['title']} ({article['source']})")

    elif sentiment == "Neutral":
        st.warning(f"{article['title']} ({article['source']})")

    else:
        st.success(f"{article['title']} ({article['source']})")



# ─────────────────────────────────────────────
# ⑤ SCENARIO SIMULATOR
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

# ── Core calculations (unchanged) ──
sim_score = (
    sim_inflation    * 0.3 +
    sim_unemployment * 0.4 +
    sim_sp500_drop   * 0.1 +
    (100 - sim_confidence) * 0.2
)
prob = min(100, sim_score * 10)
health_score = 100 - prob

# ─────────────────────────────────────────────
# ⑥ RESULTS ROW — Score | Gauge | Health
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
    gauge_color = "#ff3b30" if prob > 70 else "#ff9500" if prob > 40 else "#00d4aa"
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob,
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
            threshold=dict(line=dict(color="white", width=2), thickness=0.75, value=prob)
        )
    ))
    gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk", color="white"),
        height=260,
        margin=dict(l=30, r=30, t=40, b=10)
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
# ⑦ FORECAST CHART
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">🔮 Predictive Forecast</p>', unsafe_allow_html=True)

future       = [sim_score + i * 0.3 for i in range(6)]
future_probs = [min(100, s * 10) for s in future]
months       = ["Month 1", "Month 2", "Month 3", "Month 4", "Month 5", "Month 6"]

st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown('<p class="section-title">6-Month Economic Risk Forecast</p>', unsafe_allow_html=True)

fig_forecast = go.Figure()

fig_forecast.add_trace(go.Bar(
    x=months, y=future,
    name="Stress Score",
    marker=dict(
        color=future,
        colorscale=[[0, "rgba(0,212,170,0.7)"], [0.5, "rgba(0,122,255,0.7)"], [1, "rgba(255,59,48,0.7)"]],
        opacity=0.75
    ),
    yaxis="y",
    hovertemplate="<b>%{x}</b><br>Stress: %{y:.2f}<extra></extra>"
))

fig_forecast.add_trace(go.Scatter(
    x=months, y=future_probs,
    name="Recession Probability %",
    mode="lines+markers",
    line=dict(color="#ff9500", width=2.5, shape="spline"),
    marker=dict(size=8, color="#ff9500", line=dict(color="#050d1a", width=2)),
    yaxis="y2",
    hovertemplate="<b>%{x}</b><br>Prob: %{y:.1f}%<extra></extra>"
))

# ✅ FIX: avoid duplicate legend/xaxis/yaxis key conflicts by using targeted update methods
fig_forecast.update_layout(
    **CHART_LAYOUT,
    height=340,
    barmode="group",
    yaxis2=dict(
        title="Recession Probability (%)",
        overlaying="y", side="right",
        range=[0, 100],
        gridcolor="rgba(0,0,0,0)",
        tickfont=dict(size=11, color="rgba(255,149,0,0.7)")
    )
)
fig_forecast.update_yaxes(title_text="Stress Score", selector=dict(overlaying=None))
fig_forecast.update_layout(legend=dict(
    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
    bgcolor="rgba(255,255,255,0.04)",
    bordercolor="rgba(255,255,255,0.08)",
    borderwidth=1
))

st.plotly_chart(fig_forecast, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ⑧ AI INSIGHTS + POLICY
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
        insights.append(("warn", "🔥", "Elevated inflation above 6% increases economic instability. Purchasing power erosion may suppress consumer spending in near-term."))
    if sim_sp500_drop > 15:
        insights.append(("warn", "📉", "Sharp stock market declines of this magnitude often precede recessions. Equity sell-offs typically signal deteriorating growth expectations."))
    if sim_confidence < 80:
        insights.append(("info", "😟", "Low consumer confidence indicates potential economic slowdown. Household spending — ~70% of GDP — may contract significantly."))
    if not insights:
        insights.append(("success", "✅", "All simulated indicators are within stable ranges. Current economic parameters suggest low recession risk for the forecast horizon."))

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

    if prob > 70:
        policy_icon   = "🏦"
        policy_text   = "<strong>Immediate monetary intervention recommended.</strong> Central banks should consider easing monetary policy through rate reductions. Fiscal stimulus and enhanced unemployment support may be warranted to prevent economic contraction."
        warning_cls   = "high"
        warning_label = "🔴 SEVERE RECESSION RISK"
    elif prob > 40:
        policy_icon   = "📊"
        policy_text   = "<strong>Elevated vigilance required.</strong> Monitor economic indicators closely. Prepare contingency frameworks for rapid monetary response. Watch for leading indicators of deterioration across labor and credit markets."
        warning_cls   = "medium"
        warning_label = "🟡 MODERATE ECONOMIC RISK"
    else:
        policy_icon   = "🌿"
        policy_text   = "<strong>Conditions appear stable.</strong> Maintain current monetary policy stance. Continue routine macroprudential monitoring. Focus on structural improvements to long-term economic resilience."
        warning_cls   = "low"
        warning_label = "🟢 STABLE CONDITIONS"

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


 # ─────────────────────────────────────────────
# AI ECONOMIST CHAT
# ─────────────────────────────────────────────

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("## 🤖 AI Economist Assistant")

question = st.text_input("Ask the AI Economist")

if question:

    q = question.lower()

    # detect country automatically
    match = re.search(r"will (.+?) enter recession", q)

    if match:
        country = match.group(1).title()
    else:
        country = "Global Economy"

    if "recession" in q:

        if prob > 60:
            risk = "high"
        elif prob > 30:
            risk = "moderate"
        else:
            risk = "low"

        answer = f"""
Based on current macroeconomic indicators and financial market conditions,
the estimated recession risk for **{country}** appears **{risk}**.

This analysis considers inflation trends, labor market conditions,
and financial market volatility.
"""

    elif "inflation" in q:

        answer = f"""
Current US inflation level is **{inflation_live:.2f}**.
Persistent inflation can increase recession risk if monetary policy tightens.
"""

    elif "unemployment" in q:

        answer = f"""
Current US unemployment rate is **{unemployment_live:.2f}%**.
Labor market deterioration is one of the strongest recession indicators.
"""

    else:

        answer = """
AI analysis suggests monitoring macroeconomic indicators including
inflation, unemployment, consumer confidence, and financial markets.
"""

    st.info(answer)


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
