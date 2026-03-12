import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
from pathlib import Path
import os

st.set_page_config(page_title="Injury Risk Co-Pilot", layout="wide")

st.title("🏋️ AI Injury Risk Co-Pilot")
st.markdown("**Enter daily data → Get risk scores + suggestions**")

# 🔎 Simple explainer at the top
with st.expander("ℹ️ How to read the numbers", expanded=True):
    st.markdown("""
**Workout Intensity (0–1000)**  
• Estimates how hard the whole session was (minutes × how hard it felt on a 1–10 scale).  
• 200 = easy / recovery, 400 = normal practice, 600 = game, 800+ = very hard or double day.

**Hip Mobility (0–100%)**  
• How freely your hips can move in four quick tests:  
  1) Knee‑to‑chest while lying on your back (does the thigh get close to the chest, other leg stays flat?).  
  2) Figure‑4 stretch lying on your back (ankle on opposite knee, does the bent knee drop toward the floor?).  
  3) Seated 90/90 hip switch (can you switch sides smoothly without the back rounding?).  
  4) Body‑weight squat (can you reach at least thigh‑parallel with knees tracking over toes and no big shift?).  
• 70–100 = great, 40–60 = okay, 25–40 = warning, <25 = very stiff or painful.

**Risk Score (%)**  
• Chance of any injury in the next 7 days if training keeps looking like this.

**Status**  
• LOW = train normally.  
• MODERATE = reduce training amount 15–20% and monitor.  
• HIGH = avoid high‑intensity work, focus on recovery.
""")

# Check if model exists
model_path = "models/workload_xgb.json"
if not os.path.exists(model_path):
    st.error(f"❌ Model not found: {model_path}")
    st.info("Run `python src/model_workload.py` first to train models.")
    st.stop()

# Load model
@st.cache_resource
def load_model():
    model = xgb.Booster()
    model.load_model(model_path)
    return model

model = load_model()

# Sidebar inputs
st.sidebar.header("📊 Daily Inputs")
game_workload = st.sidebar.number_input("Workout Intensity Today (0–1000)", 0.0, 1000.0, 200.0, step=10.0)
sleep_hours = st.sidebar.slider("Sleep Hours Last Night", 0, 12, 7)
hip_mobility = st.sidebar.slider("Hip Mobility (0–100%)", 0, 100, 50)

st.sidebar.header("📈 Recent Averages")
acute_load = st.sidebar.number_input("Average Workout Intensity Last 7 Days", 0.0, 1000.0, 150.0, step=10.0)
chronic_load = st.sidebar.number_input("Average Workout Intensity Last 28 Days", 0.0, 1000.0, 140.0, step=10.0)
hip_mob_28d = st.sidebar.number_input("Average Hip Mobility Last 28 Days", 0, 100, 55)

# 🔧 Workload guide for non-experts
st.sidebar.markdown("""
**🔧 Workout Intensity Guide (Minutes × Effort 1–10)**

| Intensity | Minutes | Effort (1–10) | Example |
|-----------|---------|---------------|---------|
| **200**   | ~60 min | 3–4           | Easy skills or recovery |
| **400**   | ~80 min | 5             | Normal practice         |
| **600**   | ~90 min | 7             | Game‑like session       |
| **800+**  | 100+min | 8–10          | Very hard or double day |

**Effort scale:** 1 = resting · 5 = moderate · 10 = all‑out
""")

# Predict button
if st.button("🚨 Predict Risk", type="primary"):
    # Compute features exactly as model expects
    acwr = acute_load / max(chronic_load, 1e-6)
    acwr_spike_high = 1 if acwr > 1.5 else 0
    acwr_spike_very_high = 1 if acwr > 2.0 else 0
    hip_pct_change = (hip_mobility - hip_mob_28d) / max(hip_mob_28d, 1e-6) * 100

    input_df = pd.DataFrame({
        "game_workload": [game_workload],
        "acute_load_7d": [acute_load],
        "chronic_load_28d": [chronic_load],
        "acwr": [acwr],
        "acwr_spike_high": [acwr_spike_high],
        "acwr_spike_very_high": [acwr_spike_very_high],
        "hip_mobility": [hip_mobility],
        "hip_mobility_7d_mean": [hip_mobility],  # simple approx
        "hip_mobility_28d_mean": [hip_mob_28d],
        "hip_mobility_pct_change": [hip_pct_change],
    })

    # Predict
    dmatrix = xgb.DMatrix(input_df)
    risk_prob = model.predict(dmatrix)[0]

    # 🦵 Hip crisis safety net (clinical override)
    hip_crisis = (hip_mobility < 25) or (hip_pct_change < -60)
    if hip_crisis:
        risk_prob_original = risk_prob
        risk_prob = max(risk_prob, 0.15)  # Force HIGH RISK floor
        st.error(
            f"🦵 **HIP CRISIS OVERRIDE** | "
            f"Model: {risk_prob_original:.1%} → **HIGH RISK** | "
            f"Hip: {hip_mobility}% | Δ: {hip_pct_change:.0f}%"
        )

    # 🔥 High-load streak safety net
    load_stress = (game_workload > 600) and (acute_load > 450)
    if load_stress:
        prev = risk_prob
        risk_prob = max(risk_prob, 0.12)
        st.warning(
            f"⚠️ **HIGH LOAD STREAK DETECTED** | "
            f"Today: {game_workload:.0f} | 7‑day Avg: {acute_load:.0f} | "
            f"Risk: {prev:.1%} → {risk_prob:.1%}"
        )

    # Display results
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Risk Score", f"{risk_prob:.1%}")
    with col2:
        status = "HIGH RISK" if risk_prob > 0.10 else "Moderate" if risk_prob > 0.05 else "LOW RISK"
        color = "inverse" if risk_prob > 0.10 else "normal"
        st.metric("Status", status, delta=None, delta_color=color)
    with col3:
        st.metric("Workload Ratio (7‑day / 28‑day)", f"{acwr:.2f}")
    with col4:
        st.metric("Hip Change vs 28‑day Avg", f"{hip_pct_change:.0f}%")

    st.subheader("💡 Workout Recommendations")

    if risk_prob > 0.10:
        st.error("""
**HIGH RISK WEEK**  
❌ Skip high‑intensity work  
✅ Light mobility or active recovery  
⚠️ Monitor sleep and soreness closely
""")
    elif risk_prob > 0.05:
        st.warning("""
**MODERATE RISK**  
⚠️ Reduce training amount 15–20%  
✅ Extra warm‑up and cool‑down  
👀 Watch for fatigue signals
""")
    else:
        st.success("""
**LOW RISK**  
✅ Full workout is OK  
💪 Increase intensity if feeling good  
📊 Keep tracking each day
""")

st.markdown("---")
st.caption(
    "**XGBoost | Workload AUC 0.83 + Hip Safety Net** | "
    "[GitHub](https://github.com/JNusink/athlete-injury-workload-prediction)"
)