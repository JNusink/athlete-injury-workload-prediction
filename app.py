import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
from pathlib import Path
import os

st.set_page_config(page_title="Injury Risk Co-Pilot", layout="wide")

st.title("🏋️ AI Injury Risk Co-Pilot")
st.markdown("**Enter daily data → Get risk scores + suggestions**")

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
game_workload = st.sidebar.number_input("Game Workload", 0.0, 1000.0, 200.0, step=10.0)
sleep_hours = st.sidebar.slider("Sleep Hours", 0, 12, 7)
hip_mobility = st.sidebar.slider("Hip Mobility (0-100)", 0, 100, 50)

st.sidebar.header("📈 Recent Averages")
acute_load = st.sidebar.number_input("Avg Workload Last 7 Days", 0.0, 1000.0, 150.0, step=10.0)
chronic_load = st.sidebar.number_input("Avg Workload Last 28 Days", 0.0, 1000.0, 140.0, step=10.0)
hip_mob_28d = st.sidebar.number_input("Avg Hip Mobility Last 28 Days", 0, 100, 55)

# Predict button
if st.button("🚨 Predict Risk", type="primary"):
    # Compute features exactly as model expects
    acwr = acute_load / max(chronic_load, 1e-6)
    acwr_spike_high = 1 if acwr > 1.5 else 0
    acwr_spike_very_high = 1 if acwr > 2.0 else 0
    hip_pct_change = (hip_mobility - hip_mob_28d) / max(hip_mob_28d, 1e-6) * 100

    input_df = pd.DataFrame({
        'game_workload': [game_workload],
        'acute_load_7d': [acute_load],
        'chronic_load_28d': [chronic_load],
        'acwr': [acwr],
        'acwr_spike_high': [acwr_spike_high],
        'acwr_spike_very_high': [acwr_spike_very_high],
        'hip_mobility': [hip_mobility],
        'hip_mobility_7d_mean': [hip_mobility],  # approximate
        'hip_mobility_28d_mean': [hip_mob_28d],
        'hip_mobility_pct_change': [hip_pct_change]
    })

    # Predict
    dmatrix = xgb.DMatrix(input_df)
    risk_prob = model.predict(dmatrix)[0]

    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Risk Score", f"{risk_prob:.1%}")
    with col2:
        color = "inverse" if risk_prob > 0.10 else "normal"
        st.metric("Status", "HIGH RISK" if risk_prob > 0.10 else "Moderate" if risk_prob > 0.05 else "LOW RISK", 
                  delta=None, delta_color=color)
    with col3:
        st.metric("ACWR", f"{acwr:.2f}", delta=None)

    st.subheader("💡 Workout Recommendations")
    
    if risk_prob > 0.10:
        st.error("""
        **HIGH RISK WEEK**  
        ❌ Skip high‑intensity work  
        ✅ Light mobility or active recovery  
        ⚠️ Monitor sleep + soreness closely
        """)
    elif risk_prob > 0.05:
        st.warning("""
        **MODERATE RISK**  
        ⚠️ Reduce volume 15–20%  
        ✅ Extra warm‑up + cooldown  
        👀 Watch for fatigue signals
        """)
    else:
        st.success("""
        **LOW RISK**  
        ✅ Full workout OK  
        💪 Push intensity if feeling good  
        📊 Continue monitoring
        """)

st.markdown("---")
st.caption("**XGBoost | Workload AUC 0.83** | [GitHub](https://github.com/yourusername/sports-injury-ai-copilot)")