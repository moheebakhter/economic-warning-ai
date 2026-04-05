import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="AI Economic Early Warning System",
    page_icon="📉",
    layout="wide"
)

st.title("AI Economic Early Warning System")

# ----------------------------
# Sample Economic Data
# ----------------------------

data = {
    "inflation":[2.1,2.3,3.0,4.5,5.2,6.1,7.4],
    "unemployment":[4.1,4.3,5.0,6.1,7.2,7.8,8.2],
    "sp500":[4200,4100,3900,3700,3500,3300,3100],
    "consumer_confidence":[110,105,100,95,90,85,80],
    "stress_score":[0.27,1.43,2.96,4.89,6.58,8.13,9.72]
}

df = pd.DataFrame(data)

# ----------------------------
# KPI Economic Cards
# ----------------------------

st.subheader("Key Economic Indicators")

col1,col2,col3 = st.columns(3)

col1.metric("Inflation", "3.4%")
col2.metric("Unemployment", "3.8%")
col3.metric("S&P500", "5100")


# ----------------------------
# Dataset
# ----------------------------

st.subheader("Economic Dataset")
st.dataframe(df)

# ----------------------------
# Stress Trend
# ----------------------------

st.subheader("Economic Stress Trend")

fig, ax = plt.subplots()
ax.plot(df["stress_score"], marker="o")
ax.set_xlabel("Time")
ax.set_ylabel("Stress Score")

st.pyplot(fig)

# ----------------------------
# Stress vs Unemployment
# ----------------------------

st.subheader("Economic Stress vs Unemployment")

fig2, ax2 = plt.subplots()

ax2.scatter(df["unemployment"], df["stress_score"])

ax2.set_xlabel("Unemployment")
ax2.set_ylabel("Stress Score")

st.pyplot(fig2)

# ----------------------------
# Risk Level Table
# ----------------------------

st.subheader("Recession Risk Level")

def risk_level(score):
    if score > 6:
        return "High Risk"
    elif score > 4:
        return "Moderate Risk"
    else:
        return "Low Risk"

df["risk_level"] = df["stress_score"].apply(risk_level)

st.dataframe(df[["stress_score","risk_level"]])

# ----------------------------
# Feature Importance
# ----------------------------

st.subheader("Feature Importance")

features = {
    "sp500":0.30,
    "inflation":0.27,
    "unemployment":0.24,
    "consumer_confidence":0.18
}

fig3, ax3 = plt.subplots()

ax3.bar(features.keys(),features.values())

ax3.set_ylabel("Importance")
ax3.set_title("Economic Indicators Impact on Recession")

st.pyplot(fig3)

# ----------------------------
# Current Risk
# ----------------------------

latest_score = df["stress_score"].iloc[-1]

st.subheader("Current Economic Risk")

if latest_score > 6:
    st.error("High Recession Risk")
elif latest_score > 4:
    st.warning("Moderate Risk")
else:
    st.success("Low Risk")

# ----------------------------
# Scenario Simulator
# ----------------------------

st.header("Economic Scenario Simulator")

sim_inflation = st.slider("Inflation Rate (%)",0.0,15.0,5.0)

sim_unemployment = st.slider("Unemployment Rate (%)",0.0,15.0,6.0)

sim_sp500_drop = st.slider("Stock Market Drop (%)",0.0,50.0,10.0)

sim_confidence = st.slider("Consumer Confidence",50,120,90)

sim_score = (
    sim_inflation*0.3 +
    sim_unemployment*0.4 +
    sim_sp500_drop*0.1 +
    (100-sim_confidence)*0.2
)

st.subheader("Simulated Economic Stress Score")

st.write(round(sim_score,2))

# ----------------------------
# Recession Probability
# ----------------------------

prob = min(100, sim_score*10)

st.subheader("Predicted Recession Probability")

st.progress(int(prob))

st.write(str(round(prob,1))+"% chance of recession")

# ----------------------------
# Gauge Meter
# ----------------------------

st.subheader("Global Recession Risk Gauge")

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prob,
    title={"text":"Recession Probability %"},
    gauge={
        "axis":{"range":[0,100]},
        "bar":{"color":"red"},
        "steps":[
            {"range":[0,40],"color":"green"},
            {"range":[40,70],"color":"orange"},
            {"range":[70,100],"color":"red"}
        ]
    }
))

st.plotly_chart(gauge)

# ----------------------------
# Economic Health Score
# ----------------------------

st.subheader("Economic Health Score")

health_score = 100 - prob

st.progress(int(health_score))

st.write("Economic Health:",round(health_score,1))

# ----------------------------
# Forecast
# ----------------------------

st.subheader("6 Month Economic Risk Forecast")

future = [sim_score+i*0.3 for i in range(6)]

fig4,ax4 = plt.subplots()

ax4.plot(range(1,7),future,marker="o")

ax4.set_xlabel("Months Ahead")
ax4.set_ylabel("Economic Stress")

ax4.set_title("Projected Economic Stress")

st.pyplot(fig4)

# ----------------------------
# AI Insight
# ----------------------------

st.subheader("AI Economic Insight")

if sim_unemployment > 7:
    st.write("High unemployment is a strong recession signal.")

if sim_inflation > 6:
    st.write("High inflation increases economic instability.")

if sim_sp500_drop > 15:
    st.write("Sharp stock market declines often precede recessions.")

if sim_confidence < 80:
    st.write("Low consumer confidence indicates economic slowdown.")

# ----------------------------
# AI Policy Recommendation
# ----------------------------

st.subheader("AI Policy Recommendation")

if prob > 70:
    st.write("Central banks should consider easing monetary policy.")

elif prob > 40:
    st.write("Monitor economic indicators closely.")

else:
    st.write("Economic conditions appear stable.")

# ----------------------------
# Warning Level
# ----------------------------

st.subheader("Economic Warning Level")

if prob > 70:
    st.error("Severe Recession Risk")
elif prob > 40:
    st.warning("Moderate Economic Risk")
else:
    st.success("Economic Conditions Stable")