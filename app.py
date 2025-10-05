# app.py
import streamlit as st
import pandas as pd
import io
from model import best_model, X, viral_df, recommend_content
from features import plot_top_tags, plot_category_region, plot_postday_region
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4

# ---------------------- Page Config ----------------------
st.set_page_config(page_title="Viral Video Recommender", layout="wide", page_icon="🚀")

# ---------------------- Custom CSS ----------------------
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #e0f7fa 0%, #ede7f6 100%); }
.header-title { font-size:3.5rem; font-weight:800; text-align:center; margin-bottom:0.2rem;
                background: linear-gradient(90deg, #ff8a65, #ff5252);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.header-subtitle { text-align:center; font-size:1.3rem; color:#424242; margin-bottom:2rem; }
.tag-section { display: flex; justify-content: center; flex-wrap: wrap; gap: 15px; margin-top: 15px; }
.tag-pill { background: linear-gradient(90deg, #8e24aa, #6a1b9a); color: white; padding: 10px 18px;
            border-radius: 25px; font-size: 16px; box-shadow: 0 3px 8px rgba(0,0,0,0.2); cursor: pointer; transition: all 0.3s ease; }
.tag-pill:hover { transform: translateY(-3px); background: linear-gradient(90deg, #7b1fa2, #4a148c); box-shadow: 0 4px 15px rgba(0,0,0,0.4); }
.tips-container { display: flex; flex-direction: column; gap: 20px; margin-top: 2rem; align-items:center; }
.tip-card { background: linear-gradient(135deg, #ba68c8, #8e24aa); color: white; width: 300px; padding: 15px; border-radius: 15px;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.3); text-align: center; font-size: 15px; font-weight: 500; }
.page-btn { display:inline-block; padding:10px 20px; margin:8px; background:linear-gradient(90deg,#6a1b9a,#8e24aa);
            color:white; border:none; border-radius:25px; font-weight:600; text-decoration:none; text-align:center; transition:all 0.3s; }
.page-btn:hover { background:linear-gradient(90deg,#4a148c,#6a1b9a); transform:translateY(-2px); }
.back-btn { background:#fff; color:#6a1b9a; border:2px solid #6a1b9a; border-radius:25px; padding:10px 20px; font-weight:600; }
.back-btn:hover { background:#6a1b9a; color:white; }
</style>
""", unsafe_allow_html=True)

# ---------------------- Load Data ----------------------
file_path = r"C:\Users\Vibhu\Desktop\Predicting_Viral_content_on_Social_Media4\social_media_engagement_prerossed.csv"
df_vis = pd.read_csv(file_path)
categories = sorted(viral_df["category"].dropna().unique())
regions = sorted(viral_df["region"].dropna().unique())

# ---------------------- Page Routing ----------------------
if "page" not in st.session_state: st.session_state.page = "main"
if "result" not in st.session_state: st.session_state.result = None

def go_to(page): st.session_state.page = page

# ---------------------- PDF GENERATION ----------------------
def generate_pdf(result):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = styles['Heading1']; title_style.alignment = 1
    story.append(Paragraph("🚀 Viral Video Analytics Report", title_style))
    story.append(Spacer(1, 20))

    # Summary Table
    summary_data = [
        ["Prediction", result['Prediction']],
        ["Probability", f"{result['Probability']}%"],
        ["Best Upload Hour", f"{result['Best Upload Hour']}:00"],
        ["Best Upload Day", result['Best Upload Day']],
        ["Recommended Video Length (sec)", result['Recommended Video Length (sec)']],
    ]
    summary_table = Table(summary_data, colWidths=[200, 250])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightblue), ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 11), ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8), ('GRID', (0,0), (-1,-1), 1, colors.white),
        ('BOX', (0,0), (-1,-1), 2, colors.darkblue)
    ]))
    story.append(summary_table); story.append(Spacer(1, 20))

    # Suggested Tags Table
    tags_data = [["Suggested Tags", ", ".join(result['Suggested Tags'])]]
    tags_table = Table(tags_data, colWidths=[150, 300])
    tags_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.purple), ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 11), ('GRID', (0,0), (-1,-1), 1, colors.white)
    ]))
    story.append(tags_table); story.append(Spacer(1, 20))

    # Footer
    footer_style = styles['Italic']; footer_style.alignment = 1
    story.append(Paragraph("Generated by Viral Video Recommender", footer_style))
    story.append(Paragraph(f"Report generated on: {pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')}", footer_style))
    doc.build(story)
    buffer.seek(0)
    return buffer

# ---------------------- MAIN PAGE ----------------------
if st.session_state.page == "main":
    st.markdown("<h1 class='header-title'>🚀 Viral Video Recommender</h1>", unsafe_allow_html=True)
    st.markdown("<p class='header-subtitle'>Get AI-powered insights to create the next viral sensation!</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader("🎥 Video Configuration")
        category_input = st.selectbox("📂 Category", categories, key="category_select")
        region_input = st.selectbox("🌍 Region", regions, key="region_select")
        video_length_input = st.slider("⏱️ Video Length (seconds)", 30, 3600, 300, step=10, key="video_length_slider")
        publish_hour_input = st.slider("🕐 Publish Hour (0-23)", 0, 23, 12, key="publish_hour_slider")
        if st.button("🚀 Generate Recommendation", key="generate_button"):
            with st.spinner("🔮 Analyzing viral patterns..."):
                st.session_state.result = recommend_content(category_input, region_input, video_length_input, publish_hour_input)

    with col2:
        result = st.session_state.result
        if result:
            st.markdown(f"""
            <div style='background:#fff8e1;border:2px solid #ffe082;padding:1.5rem;border-radius:15px;text-align:center;'>
                <h2 style='color:#6a1b9a;'>{result["Prediction"]} ({result["Probability"]}%)</h2>
            </div>
            """, unsafe_allow_html=True)
            st.progress(result["Probability"]/100)
            st.markdown("<h3 style='text-align:center; color:#4a148c;'>🏷️ Suggested Tags</h3>", unsafe_allow_html=True)
            st.markdown("<div class='tag-section'>" + "".join([f"<span class='tag-pill'>{tag}</span>" for tag in result["Suggested Tags"]]) + "</div>", unsafe_allow_html=True)

            # --- Download Report Button ---
            st.subheader("📥 Download Analytics Report")
            col1, col2, col3 = st.columns([1,1,1])
            with col2:
                pdf_buffer = generate_pdf(result)
                st.download_button("📥 Download PDF Report", data=pdf_buffer, file_name="viral_video_report.pdf", mime="application/pdf", key="download_pdf")

            st.markdown("<br><div style='text-align:center;'>", unsafe_allow_html=True)
            st.markdown("<h3>Select a Section</h3>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: st.button("💡 Pro Tips", on_click=lambda: go_to("tips"), key="pro_tips_btn")
            with c2: st.button("📊 Historical Insights", on_click=lambda: go_to("insights"), key="historical_btn")
            st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- PRO TIPS PAGE ----------------------
elif st.session_state.page == "tips":
    result = st.session_state.result
    st.markdown("<h1 class='header-title'>💡 Pro Tips</h1>", unsafe_allow_html=True)
    st.markdown("<p class='header-subtitle'>Follow these expert insights to maximize your content’s performance.</p>", unsafe_allow_html=True)
    
    if result:
        st.markdown("<div class='tips-container'>", unsafe_allow_html=True)
        st.markdown(f"<div class='tip-card'>📅 Upload at <b>{result['Best Upload Hour']}:00</b> on <b>{result['Best Upload Day']}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='tip-card'>🎬 Keep your video around <b>{result['Recommended Video Length (sec)']} seconds</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='tip-card'>🏷️ Use trending tags like <b>{', '.join(result['Suggested Tags'])}</b></div>", unsafe_allow_html=True)
        st.markdown("<div class='tip-card'>📈 Stay consistent with uploads to build audience trust and engagement</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.button("⬅️ Back to Main", on_click=lambda: go_to("main"), key="back_main_tips")

# ---------------------- INSIGHTS PAGE ----------------------
elif st.session_state.page == "insights":
    result = st.session_state.result
    st.markdown("<h1 class='header-title'>📊 Historical Data Insights</h1>", unsafe_allow_html=True)
    st.markdown("<p class='header-subtitle'>Visualize engagement patterns and discover what drives virality.</p>", unsafe_allow_html=True)
    
    if result:
        with st.expander("📈 Top Performing Tags", expanded=True):
            plot_top_tags(df_vis, category=result.get('Category'), region=result.get('Region'))
        with st.expander("🌍 Category Performance by Region", expanded=False):
            plot_category_region(df_vis, category=result.get('Category'))
        with st.expander("📅 Best Posting Days Analysis", expanded=False):
            plot_postday_region(df_vis, category=result.get('Category'), region=result.get('Region'))

    st.button("⬅️ Back to Main", on_click=lambda: go_to("main"), key="back_main_insights")
