# app.py
import streamlit as st
import pandas as pd
import numpy as np
import calendar
from model import best_model, X, viral_df, recommend_content

st.set_page_config(page_title="Viral Video Recommender", layout="wide", page_icon="📈")
st.title("📈 Viral Video Recommendation System")
st.markdown("Get AI-powered insights for creating viral videos based on historical trends!")

# ----------------------
# Streamlit frontend inputs
# ----------------------
st.subheader("🎯 Enter Video Details")
categories = sorted(viral_df["category"].dropna().unique())
regions = sorted(viral_df["region"].dropna().unique())

col1, col2 = st.columns(2)
with col1:
    category_input = st.selectbox("Category", categories)
with col2:
    region_input = st.selectbox("Region", regions)

col3, col4 = st.columns(2)
with col3:
    video_length_input = st.slider("Video Length (seconds)", 30, 3600, 300, step=10)
with col4:
    publish_hour_input = st.slider("Publish Hour (0-23)", 0, 23, 12)

# ----------------------
# Show Recommendation
# ----------------------
if st.button("Get Recommendation"):
    result = recommend_content(category_input, region_input, video_length_input, publish_hour_input)
    st.success("✅ Recommendation Ready!")

    # ---------- Top Metrics ----------
    st.markdown("### 📊 Key Insights")
    col1, col2, col3 = st.columns(3)
    
    # Prediction with color
    if result["Prediction"] == "Viral":
        col1.metric("Prediction", result["Prediction"], f"{result['Probability']}% Probability", delta_color="normal")
    else:
        col1.metric("Prediction", result["Prediction"], f"{result['Probability']}% Probability", delta_color="inverse")

    col2.metric("Best Upload Hour", f"{result['Best Upload Hour']}:00")
    col3.metric("Best Upload Day", result["Best Upload Day"])

    # Probability progress bar
    st.markdown("### 🎯 Viral Probability")
    st.progress(result["Probability"] / 100)

    # Recommended length
    st.markdown(f"### ⏱ Recommended Video Length: {result['Recommended Video Length (sec)']} seconds")
    st.info(f"Try to keep your video around {result['Recommended Video Length (sec)']} sec for better engagement.")

    # Suggested tags as clickable chips
    st.markdown("### 🏷 Suggested Tags")
    tags_html = " ".join([f"<span style='background-color:#FFD700;color:#000;padding:5px;margin:3px;border-radius:5px;'>{tag}</span>" for tag in result["Suggested Tags"]])
    st.markdown(tags_html, unsafe_allow_html=True)

    # Optional tips
    st.markdown("""
    ### 💡 Tips:
    - Publish at the suggested hour and day for higher reach.
    - Use the suggested tags to increase discoverability.
    - Video length close to recommended duration tends to retain viewers longer.
    """)
