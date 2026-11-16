"""
app.py
------
Streamlit dashboard for the Soil Carbon Agentic AI.
Designed for LinkedIn demo + portfolio showcase.

Features:
- Interactive Folium map with SOC heatmap
- Real-time agent reasoning trace
- Field-level recommendations
- Carbon credit potential calculator
- Exportable report
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import json
import time

from src.soc_model import generate_synthetic_training_data, SOCPredictor, PracticeClassifier
from src.indices import compute_indices_df, add_practice_labels, get_carbon_potential
from src.agent import run_mock_agent


# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="🌱 Soil Carbon Agent | GEO AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background-color: #0e1117; }
    
    .metric-card {
        background: linear-gradient(135deg, #1a2634, #0d1b2a);
        border: 1px solid #2d4a6e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #4ecdc4;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #8899aa;
        margin-top: 4px;
    }
    
    .agent-step {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-left: 4px solid #4ecdc4;
        border-radius: 8px;
        padding: 16px;
        margin: 10px 0;
    }
    
    .priority-high   { color: #ff6b6b; font-weight: bold; }
    .priority-medium { color: #ffd93d; font-weight: bold; }
    .priority-low    { color: #6bcb77; font-weight: bold; }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #4ecdc4, #556270);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    
    district = st.selectbox(
        "📍 Select District",
        ["Wardha", "Yavatmal", "Amravati", "Nashik"],
        index=0
    )
    
    n_fields = st.slider("📊 Number of Field Samples", 50, 300, 150, 25)
    
    season = st.selectbox(
        "🌾 Season",
        ["Rabi 2023-24 (Oct-Mar)", "Kharif 2023 (Jun-Sep)"],
        index=0
    )
    
    use_real_gee = st.toggle("🛰️ Use Real GEE Data", value=False,
                              help="Toggle ON to use real Sentinel-2 data via GEE (requires auth)")
    
    st.markdown("---")
    st.markdown("**Data Sources:**")
    st.markdown("🛰️ Sentinel-2 SR (ESA Copernicus)")
    st.markdown("🌱 SoilGrids 2.0 (ISRIC)")
    st.markdown("🤖 LangGraph Agent Pipeline")
    st.markdown("---")
    st.markdown("**Built by:** GEO AI Developer")
    st.markdown("**Stack:** GEE · Scikit-learn · LangGraph · Streamlit")


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown('<p class="header-title">🌱 Soil Carbon Agentic AI</p>', unsafe_allow_html=True)
st.markdown(f"#### Autonomous SOC Monitoring & Regenerative Farming Advisor | **{district}, Maharashtra**")
st.markdown("*Powered by Sentinel-2 · SoilGrids · LangGraph Multi-Agent Pipeline*")
st.markdown("---")


# ─────────────────────────────────────────────
# Data Loading & Model Training
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_and_process_data(district: str, n_fields: int, seed: int = 42):
    """Load, process, and predict on field data."""
    np.random.seed(seed)
    
    district_coords = {
        "Wardha":   {"lat_range": (20.40, 21.10), "lon_range": (78.35, 79.10)},
        "Yavatmal": {"lat_range": (19.80, 20.80), "lon_range": (77.50, 78.60)},
        "Amravati": {"lat_range": (20.50, 21.30), "lon_range": (77.20, 78.00)},
        "Nashik":   {"lat_range": (19.50, 20.50), "lon_range": (73.50, 74.50)},
    }
    coords = district_coords.get(district, district_coords["Wardha"])
    
    # Generate data
    df = generate_synthetic_training_data(max(n_fields * 4, 1000))
    df["latitude"]  = np.random.uniform(*coords["lat_range"], len(df))
    df["longitude"] = np.random.uniform(*coords["lon_range"], len(df))
    
    df = compute_indices_df(df)
    df = add_practice_labels(df)
    
    # Train models
    soc_model = SOCPredictor()
    soc_model.train(df)
    
    prac_clf = PracticeClassifier()
    prac_clf.train(df)
    
    # Inference on sample
    sample_df = df.sample(n_fields, random_state=42).reset_index(drop=True)
    sample_df = soc_model.predict(sample_df)
    sample_df = prac_clf.predict_proba_df(sample_df)
    
    return sample_df, soc_model.metrics


with st.spinner("🛰️ Processing satellite imagery and training models..."):
    df, model_metrics = load_and_process_data(district, n_fields)

st.success(f"✅ Loaded {len(df)} field samples | Model R² = {model_metrics['R2']} | RMSE = {model_metrics['RMSE']} g/kg")


# ─────────────────────────────────────────────
# KPI Metrics Row
# ─────────────────────────────────────────────

avg_soc      = round(df["SOC_predicted"].mean(), 1)
pct_degraded = round((df["PredictedPractice"] == "Conventional Tilled / Bare Soil").mean() * 100, 1)
pct_healthy  = round((df["SOC_predicted"] >= 15).mean() * 100, 1)
carbon_gap   = round((20 - avg_soc) * len(df) * 0.3 / 1000, 2)  # rough tC estimate

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🌱 Avg SOC", f"{avg_soc} g/kg",
              delta=f"{round(avg_soc - 20, 1)} vs healthy",
              delta_color="inverse")

with col2:
    st.metric("⚠️ Degraded Fields", f"{pct_degraded}%",
              delta="Need urgent intervention",
              delta_color="inverse")

with col3:
    st.metric("✅ Healthy Fields", f"{pct_healthy}%",
              delta="SOC > 15 g/kg")

with col4:
    st.metric("💰 Carbon Gap", f"{carbon_gap} tC",
              delta="Sequestration potential",
              delta_color="normal")

st.markdown("---")


# ─────────────────────────────────────────────
# Main Layout: Map + Charts
# ─────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🗺️ Interactive Map", "📊 Analysis Charts", "🤖 Agent Reasoning"])


# ───────── TAB 1: MAP ─────────────────────────

with tab1:
    st.markdown("### 🛰️ Field-Level SOC Heatmap — " + district + ", Maharashtra")
    st.caption("Color = SOC level (green=high, red=low) | Circle size = confidence")

    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="CartoDB dark_matter"
    )

    # Color scale for SOC
    def soc_color(soc):
        if soc < 5:   return "#d73027"   # deep red
        elif soc < 8: return "#fc8d59"   # orange-red
        elif soc < 12:return "#fee090"   # yellow
        elif soc < 16:return "#91cf60"   # light green
        elif soc < 22:return "#1a9850"   # green
        else:         return "#006837"   # dark green

    # Add field markers
    for _, row in df.iterrows():
        color = soc_color(row["SOC_predicted"])
        radius = max(4, row.get("PracticeConfidence", 70) / 20)

        popup_html = f"""
        <div style='font-family:monospace; width:220px;'>
        <b>📍 Field Report</b><br>
        <hr>
        <b>SOC:</b> {row['SOC_predicted']} g/kg ({row['SOC_category']})<br>
        <b>Practice:</b> {row['PredictedPractice']}<br>
        <b>Confidence:</b> {row.get('PracticeConfidence', 'N/A')}%<br>
        <b>NDVI:</b> {round(row.get('NDVI', 0), 3)}<br>
        <b>BSI:</b> {round(row.get('BSI', 0), 3)}<br>
        <hr>
        <b>Carbon Credit:</b> ₹{get_carbon_potential(row['PredictedPractice'])['credit_potential_per_ha']:,.0f}/ha/yr
        </div>
        """

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"SOC: {row['SOC_predicted']} g/kg | {row['PredictedPractice'][:20]}"
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:9999;
                background:rgba(0,0,0,0.8); padding:12px; border-radius:8px;
                font-family:monospace; color:white; font-size:12px; border:1px solid #333;">
    <b>SOC Level (g/kg)</b><br>
    <span style="color:#d73027">●</span> &lt; 5 (Critical)<br>
    <span style="color:#fc8d59">●</span> 5–8 (Very Low)<br>
    <span style="color:#fee090">●</span> 8–12 (Low)<br>
    <span style="color:#91cf60">●</span> 12–16 (Moderate)<br>
    <span style="color:#1a9850">●</span> 16–22 (Good)<br>
    <span style="color:#006837">●</span> &gt; 22 (Excellent)
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, width=None, height=550)


# ───────── TAB 2: CHARTS ──────────────────────

with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🌾 Land Practice Distribution")
        practice_counts = df["PredictedPractice"].value_counts().reset_index()
        practice_counts.columns = ["Practice", "Count"]

        fig_pie = px.pie(
            practice_counts,
            names="Practice",
            values="Count",
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.45
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            showlegend=True,
            height=350
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown("#### 📊 SOC Distribution")
        fig_hist = px.histogram(
            df, x="SOC_predicted",
            nbins=30,
            color_discrete_sequence=["#4ecdc4"],
            labels={"SOC_predicted": "SOC (g/kg)"}
        )
        fig_hist.add_vline(x=20, line_dash="dash", line_color="#ff6b6b",
                           annotation_text="Healthy threshold (20 g/kg)")
        fig_hist.add_vline(x=avg_soc, line_dash="dot", line_color="#ffd93d",
                           annotation_text=f"Study mean ({avg_soc})")
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(14,17,23,0.8)",
            font_color="white",
            height=350
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # SOC by Practice Box Plot
    st.markdown("#### 🎻 SOC by Land Practice")
    fig_box = px.box(
        df, x="PredictedPractice", y="SOC_predicted",
        color="PredictedPractice",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        labels={"SOC_predicted": "SOC (g/kg)", "PredictedPractice": "Land Practice"}
    )
    fig_box.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.8)",
        font_color="white",
        height=380,
        showlegend=False,
        xaxis_tickangle=-30
    )
    fig_box.add_hline(y=20, line_dash="dash", line_color="#ff6b6b",
                      annotation_text="Healthy SOC threshold")
    st.plotly_chart(fig_box, use_container_width=True)


# ───────── TAB 3: AGENT REASONING ─────────────

with tab3:
    st.markdown("### 🤖 Agentic AI Reasoning Pipeline")
    st.markdown("*Watch the AI agent perceive → analyze → diagnose → recommend → report*")
    st.caption("This is the core agentic loop — 5 specialized nodes reasoning in sequence.")

    col_run, col_info = st.columns([1, 3])
    with col_run:
        run_agent_btn = st.button("▶️  Run Agent Pipeline", type="primary", use_container_width=True)
    with col_info:
        st.info("💡 The agent will reason step-by-step across 5 nodes: Perception → Analysis → Diagnosis → Recommendation → Report")

    if run_agent_btn or "agent_result" in st.session_state:

        if run_agent_btn:
            field_data_list = df.to_dict("records")

            with st.status("🤖 Agent pipeline running...", expanded=True) as status:
                st.write("🔍 Node 1: Perception — reading satellite data...")
                time.sleep(0.5)
                st.write("📊 Node 2: Analysis — interpreting SOC patterns...")
                time.sleep(0.5)
                st.write("🩺 Node 3: Diagnosis — identifying critical fields...")
                time.sleep(0.5)
                st.write("💡 Node 4: Recommendation — generating interventions...")
                time.sleep(0.5)
                st.write("📋 Node 5: Report — compiling executive summary...")

                result = run_mock_agent(field_data_list, f"{district}, Maharashtra")
                st.session_state["agent_result"] = result
                status.update(label="✅ Agent pipeline complete!", state="complete")

        result = st.session_state["agent_result"]

        # Display reasoning trace
        st.markdown("---")
        st.markdown("#### 🔍 Agent Reasoning Trace")

        icons = ["🔍", "📊", "🩺", "💡", "📋"]
        node_names = ["Perception", "Analysis", "Diagnosis", "Recommendation", "Report"]
        colors = ["#4ecdc4", "#ffd93d", "#ff6b6b", "#6bcb77", "#a78bfa"]

        for i, trace in enumerate(result["reasoning_trace"]):
            with st.expander(f"{icons[i]} Node {i+1}: {node_names[i]}", expanded=(i == 0)):
                st.markdown(f"""
                <div style="border-left: 4px solid {colors[i]}; 
                            padding: 12px; border-radius: 4px;
                            background: rgba(0,0,0,0.3);">
                """, unsafe_allow_html=True)

                if i == 3:  # Recommendations — render as cards
                    st.markdown("**Generated Action Plan:**")
                    recs = result["recommendations"]
                    for rec in recs:
                        priority_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(rec["priority"], "⚪")
                        st.markdown(f"""
                        **{priority_color} {rec['practice']}**
                        - *{rec['description']}*
                        - Carbon benefit: `{rec['carbon_benefit']}`
                        - Season: `{rec['implementation_season']}`
                        - Credit potential: `₹{rec['credit_potential_inr_per_ha']:,}/ha/year`
                        - Timeline: `{rec['timeline']}`
                        """)
                        st.markdown("---")
                else:
                    content = trace.split(":", 1)[-1].strip() if ":" in trace else trace
                    st.markdown(content)

                st.markdown("</div>", unsafe_allow_html=True)

        # Final Report
        st.markdown("---")
        st.markdown("#### 📋 Executive Report")
        st.markdown(result["final_report"])

        # Download button
        st.download_button(
            label="📥 Download Full Report",
            data=result["final_report"],
            file_name=f"soil_carbon_report_{district.lower()}.md",
            mime="text/markdown"
        )

    else:
        # Placeholder showing agent architecture
        st.markdown("#### Agent Architecture")
        arch_html = """
        <div style="font-family: monospace; font-size: 14px; 
                    background: rgba(0,0,0,0.4); padding: 24px; 
                    border-radius: 12px; border: 1px solid #2d4a6e;
                    text-align: center;">
        <span style="color:#4ecdc4; font-size:16px;">🛰️ SATELLITE DATA (Sentinel-2)</span><br>
        <span style="color:#555">│</span><br>
        <span style="color:#4ecdc4">┌──────────────────────────┐</span><br>
        <span style="color:#fff">│  🔍 PERCEPTION NODE       │</span> <span style="color:#888">← reads imagery, computes indices</span><br>
        <span style="color:#4ecdc4">└──────────────┬───────────┘</span><br>
        <span style="color:#555">               │</span><br>
        <span style="color:#ffd93d">┌──────────────────────────┐</span><br>
        <span style="color:#fff">│  📊 ANALYSIS NODE         │</span> <span style="color:#888">← interprets SOC + practices</span><br>
        <span style="color:#ffd93d">└──────────────┬───────────┘</span><br>
        <span style="color:#555">               │</span><br>
        <span style="color:#ff6b6b">┌──────────────────────────┐</span><br>
        <span style="color:#fff">│  🩺 DIAGNOSIS NODE        │</span> <span style="color:#888">← flags critical fields</span><br>
        <span style="color:#ff6b6b">└──────────────┬───────────┘</span><br>
        <span style="color:#555">               │</span><br>
        <span style="color:#6bcb77">┌──────────────────────────┐</span><br>
        <span style="color:#fff">│  💡 RECOMMENDATION NODE   │</span> <span style="color:#888">← generates interventions</span><br>
        <span style="color:#6bcb77">└──────────────┬───────────┘</span><br>
        <span style="color:#555">               │</span><br>
        <span style="color:#a78bfa">┌──────────────────────────┐</span><br>
        <span style="color:#fff">│  📋 REPORT NODE           │</span> <span style="color:#888">← compiles executive report</span><br>
        <span style="color:#a78bfa">└──────────────────────────┘</span><br>
        </div>
        """
        st.markdown(arch_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#555; font-size:12px;">
🌱 Soil Carbon Agentic AI | Built with Sentinel-2 · SoilGrids · LangGraph · Streamlit
| Data: ESA Copernicus + ISRIC SoilGrids 2.0 | Demo uses synthetic data for illustration
</div>
""", unsafe_allow_html=True)
