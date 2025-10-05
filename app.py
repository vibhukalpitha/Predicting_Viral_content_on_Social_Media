# app.py
import streamlit as st
import pandas as pd
import numpy as np
from model import best_model, X, viral_df, recommend_content
from features import plot_top_tags, plot_category_region, plot_postday_region

# ---------------------- Page Configuration ----------------------
st.set_page_config(
    page_title="Viral Video Recommender",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="collapsed"
)

# ---------------------- Custom CSS Styling ----------------------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(to bottom, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Header styling */
    .header-title {
        color: #1a73e8;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        text-align: center;
        color: #5f6368;
        font-size: 1.3rem;
        margin-bottom: 2rem;
    }
    
    /* Input section styling */
    .input-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        border: 1px solid #e8eaed;
    }
    
    .input-section h3 {
        color: #1a73e8;
        font-size: 1.8rem;
        margin-bottom: 1.5rem;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1a73e8;
    }
    
    /* Tag styling */
    .tag-pill {
        display: inline-block;
        background: #e8f0fe;
        color: #1967d2;
        padding: 8px 16px;
        margin: 5px;
        border-radius: 20px;
        font-weight: 600;
        border: 1px solid #d2e3fc;
        transition: all 0.2s;
    }
    
    .tag-pill:hover {
        background: #d2e3fc;
        transform: translateY(-2px);
    }
    
    /* Prediction card */
    .prediction-card {
        background: #fff3e0;
        padding: 1.5rem;
        border-radius: 15px;
        color: #e65100;
        text-align: center;
        box-shadow: 0 2px 10px rgba(230, 81, 0, 0.1);
        margin: 1rem 0;
        border: 2px solid #ffcc80;
    }
    
    .prediction-viral {
        background: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #81c784;
    }
    
    /* Button styling */
    .stButton>button {
        background: #1a73e8;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 8px rgba(26, 115, 232, 0.3);
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton>button:hover {
        background: #1557b0;
        box-shadow: 0 4px 12px rgba(26, 115, 232, 0.4);
        transform: translateY(-2px);
    }
    
    /* Info boxes */
    .tip-box {
        background: #fff8e1;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #ffa726;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Progress bar customization */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #4caf50 0%, #8bc34a 100%);
    }
    
    /* Selectbox and slider labels */
    .stSelectbox label, .stSlider label {
        color: #1a73e8 !important;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Section headers */
    .section-header {
        color: #1a73e8;
        font-size: 2rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        text-align: center;
    }
    
    /* Card containers */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
        border: 1px solid #e8eaed;
        transition: all 0.3s;
    }
    
    .card:hover {
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
        transform: translateY(-3px);
    }
    
    /* Emoji enhancement */
    .emoji-large {
        font-size: 3rem;
        display: block;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Custom divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(to right, transparent, #e8eaed, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------- Header ----------------------
st.markdown("<h1 class='header-title'>🚀 Viral Video Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p class='header-subtitle'>Get AI-powered insights to create the next viral sensation!</p>", unsafe_allow_html=True)

# ---------------------- Load Dataset ----------------------
file_path = r"C:\Users\User\OneDrive\Desktop\FDM_New_Project\social_media_preprossed.csv"
df_vis = pd.read_csv(file_path)

# ---------------------- Prepare Dropdown Options ----------------------
categories = sorted(viral_df["category"].dropna().unique())
regions = sorted(viral_df["region"].dropna().unique())

# ---------------------- Main Content Layout ----------------------
col_left, col_right = st.columns([1, 2], gap="large")

# ---------------------- Left Column: User Inputs ----------------------
with col_left:
    
    st.markdown("<h3>Video Configuration</h3>", unsafe_allow_html=True)
    
    st.markdown("#### 📂 Category")
    category_input = st.selectbox("Category", categories, label_visibility="collapsed")
    
    st.markdown("#### 🌍 Region")
    region_input = st.selectbox("Region", regions, label_visibility="collapsed")
    
    st.markdown("#### ⏱️ Video Length")
    video_length_input = st.slider(
        "Video Length (seconds)", 
        30, 3600, 300, step=10,
        label_visibility="collapsed"
    )
    st.caption(f"Selected: {video_length_input} seconds ({video_length_input//60}m {video_length_input%60}s)")
    
    st.markdown("#### 🕐 Publish Hour")
    publish_hour_input = st.slider(
        "Publish Hour (0-23)", 
        0, 23, 12,
        label_visibility="collapsed"
    )
    st.caption(f"Selected: {publish_hour_input}:00")
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Generate Recommendation", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- Right Column: Results ----------------------
with col_right:
    if 'result' not in st.session_state:
        st.session_state.result = None

    if predict_btn:
        with st.spinner("🔮 Analyzing viral patterns..."):
            result = recommend_content(category_input, region_input, video_length_input, publish_hour_input)
            st.session_state.result = result

    result = st.session_state.result

    if result:
        # Prediction Banner
        prediction_class = "prediction-viral" if result["Prediction"] == "Viral" else ""
        viral_color = "#2e7d32" if result["Prediction"] == "Viral" else "#e65100"
        st.markdown(f"""
        <div class='prediction-card {prediction_class}'>
            <div class='emoji-large'>{'🔥' if result["Prediction"] == "Viral" else '📊'}</div>
            <h2 style='margin:0; color:{viral_color};'>{result["Prediction"].upper()}</h2>
            <h1 style='margin:0.5rem 0; color:{viral_color};'>{result['Probability']}%</h1>
            <p style='margin:0; color:{viral_color}; opacity:0.8;'>Viral Probability</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar
        st.progress(result["Probability"] / 100)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Key Metrics in Cards
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.markdown(f"""
            <div class='card'>
                <div style='text-align:center;'>
                    <div style='font-size:2rem;'>🕐</div>
                    <h3 style='color:#1a73e8;margin:0.5rem 0;'>{result['Best Upload Hour']}:00</h3>
                    <p style='color:#5f6368;margin:0;'>Best Upload Hour</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown(f"""
            <div class='card'>
                <div style='text-align:center;'>
                    <div style='font-size:2rem;'>📅</div>
                    <h3 style='color:#1a73e8;margin:0.5rem 0;'>{result['Best Upload Day']}</h3>
                    <p style='color:#5f6368;margin:0;'>Best Upload Day</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col3:
            st.markdown(f"""
            <div class='card'>
                <div style='text-align:center;'>
                    <div style='font-size:2rem;'>⏱️</div>
                    <h3 style='color:#1a73e8;margin:0.5rem 0;'>{result['Recommended Video Length (sec)']}s</h3>
                    <p style='color:#5f6368;margin:0;'>Optimal Length</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Suggested Tags
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🏷️ Suggested Tags")
        tags_html = "".join([f"<span class='tag-pill'>{tag}</span>" for tag in result["Suggested Tags"]])
        st.markdown(tags_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Tips Section
        st.markdown(f"""
        <div class='tip-box'>
            <h3 style='color:#f57c00;margin-top:0;'>💡 Pro Tips for Maximum Engagement</h3>
            <ul style='color:#424242;line-height:1.8;'>
                <li><strong>Timing is Everything:</strong> Upload at {result['Best Upload Hour']}:00 on {result['Best Upload Day']} for peak visibility</li>
                <li><strong>Length Matters:</strong> Keep your video around {result['Recommended Video Length (sec)']} seconds to maximize retention</li>
                <li><strong>Tag Smart:</strong> Use all suggested tags to improve discoverability across platforms</li>
                <li><strong>Consistency:</strong> Post regularly during your optimal time windows</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Welcome message when no prediction yet
        st.markdown("""
        <div class='card' style='text-align:center;padding:3rem;'>
            <div style='font-size:4rem;margin-bottom:1rem;'>🎬</div>
            <h2 style='color:#1a73e8;'>Ready to Go Viral?</h2>
            <p style='color:#5f6368;font-size:1.1rem;'>
                Configure your video details on the left and click 
                <strong>"Generate Recommendation"</strong> to get personalized insights!
            </p>
        </div>
        """, unsafe_allow_html=True)

# ---------------------- Data Insights Section ----------------------
if st.session_state.result:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-header'>📊 Historical Data Insights</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#5f6368;font-size:1.1rem;'>Explore patterns and trends from historical viral content</p>", unsafe_allow_html=True)
    
    # Charts in expandable sections
    with st.expander("📈 Top Performing Tags", expanded=True):
        plot_top_tags(df_vis, category=category_input, region=region_input)
    
    with st.expander("🌍 Category Performance by Region", expanded=False):
        plot_category_region(df_vis, category=category_input)
    
    with st.expander("📅 Best Posting Days Analysis", expanded=False):
        plot_postday_region(df_vis, category=category_input, region=region_input)

# ---------------------- Footer ----------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;color:#5f6368;padding:2rem;'>
    <p style='font-size:0.9rem;'>
         using AI-powered predictions | © 2025 Viral Video Recommender
    </p>
</div>
""", unsafe_allow_html=True)