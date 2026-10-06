"""Smart Attendance Anomaly Detector - Modern Web Interface

Owner: Arjun Kumar (btech/10433/24)
Run: streamlit run app.py

Features:
 - Three.js Inclined WebGL Canvas (Particle Embers & Geometry Core)
 - Anomaly Taxonomy & Explanation Engine
 - Student Deep-Dive Inspector & Timeline
 - Interactive Threshold Tuning & Sensitivity Control
 - Plotly Dark Theme Visualizations
 - Benchmark & Model Comparison Hub
 - Administrative CSV Report Export
"""
import io
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

from src.data_generator import generate_attendance
from src.features import build_features, FEATURE_NAMES
from src.model_isolation_forest import IsolationForestDetector
from src.model_autoencoder import AutoencoderDetector
from src.improved_model import HybridDetector
from src.explain import categorize_and_explain

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Attendance Anomaly Detector",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS: CLEAN DARK THEME & ACCENTS
# -----------------------------------------------------------------------------
APP_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Onest:wght@300;400;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

    :root {
        --bg-main: #0b0c10;
        --card-bg: #12141a;
        --border-color: #1f232d;
        --accent-red: #e0231c;
        --accent-red-glow: rgba(224, 35, 28, 0.35);
        --text-muted: #8a919e;
    }

    .stApp {
        background-color: var(--bg-main);
        color: #e2e8f0;
        font-family: 'Onest', sans-serif;
    }

    header[data-testid="stHeader"] {
        background-color: rgba(11, 12, 16, 0.85) !important;
        backdrop-filter: blur(12px);
        border-bottom: 1px solid var(--border-color);
    }
    footer { visibility: hidden; }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0d0e14 !important;
        border-right: 1px solid var(--border-color);
    }

    /* Hero Header Typography */
    .hero-header {
        font-family: 'Instrument Serif', serif;
        font-size: 3.6rem;
        font-weight: 400;
        line-height: 1.05;
        color: #ffffff;
        letter-spacing: -0.012em;
        margin-top: 4px;
    }

    .red-accent {
        color: var(--accent-red);
        text-shadow: 0 0 15px var(--accent-red-glow);
    }

    .badge-tag {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(224, 35, 28, 0.08);
        border: 1px solid rgba(224, 35, 28, 0.3);
        color: var(--accent-red);
        padding: 4px 14px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
    }

    /* Shelf Cards */
    .shelf-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 20px;
        position: relative;
        overflow: hidden;
        transition: transform 0.25s ease, border-color 0.25s ease;
    }

    .shelf-card:hover {
        border-color: rgba(224, 35, 28, 0.5);
        transform: translateY(-2px);
    }

    .card-index {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--accent-red);
        margin-bottom: 6px;
        letter-spacing: 0.1em;
    }

    .metric-value-display {
        font-family: 'Instrument Serif', serif;
        font-size: 2.2rem;
        color: #ffffff;
    }

    /* Interactive Buttons */
    div.stButton > button, div.stDownloadButton > button {
        background-color: var(--accent-red) !important;
        color: #ffffff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 8px 20px !important;
        box-shadow: 0 0 18px var(--accent-red-glow);
        transition: all 0.2s ease !important;
    }

    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background-color: #f5312a !important;
        box-shadow: 0 0 28px rgba(224, 35, 28, 0.65);
    }

    /* Tab Headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid var(--border-color);
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(18, 20, 26, 0.6);
        border: 1px solid var(--border-color);
        border-radius: 6px 6px 0 0;
        color: var(--text-muted);
        padding: 8px 18px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--card-bg) !important;
        border-color: var(--accent-red) !important;
        border-bottom: 2px solid var(--accent-red) !important;
        color: var(--accent-red) !important;
        font-weight: 600;
    }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# THREE.JS INCLINED WEBGL CANVAS (PARTICLE EMBERS & WIREFRAME CORE)
# -----------------------------------------------------------------------------
THREE_CANVAS_HTML = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { margin: 0; overflow: hidden; background-color: #0b0c10; }
        canvas { display: block; }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
<script>
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x0b0c10, 0.04);

    const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 3, 12);
    camera.rotation.x = -0.15;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    document.body.appendChild(renderer.domElement);

    // Inclined Grid Floor
    const grid = new THREE.GridHelper(50, 50, 0xe0231c, 0x1f232d);
    grid.position.y = -2;
    scene.add(grid);

    // Rising Red Embers
    const emberCount = 350;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(emberCount * 3);

    for (let i = 0; i < emberCount * 3; i += 3) {
        positions[i] = (Math.random() - 0.5) * 30;
        positions[i + 1] = Math.random() * 15 - 2;
        positions[i + 2] = (Math.random() - 0.5) * 30;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const material = new THREE.PointsMaterial({
        size: 0.07,
        color: 0xe0231c,
        transparent: true,
        opacity: 0.85
    });

    const emberParticles = new THREE.Points(geometry, material);
    scene.add(emberParticles);

    // Dynamic Geometry Node (Anomaly Core)
    const torusGeo = new THREE.TorusGeometry(2, 0.4, 12, 32);
    const wireMat = new THREE.MeshBasicMaterial({
        color: 0xe0231c,
        wireframe: true,
        transparent: true,
        opacity: 0.45
    });
    const anomalyNode = new THREE.Mesh(torusGeo, wireMat);
    anomalyNode.position.set(0, 0.5, 0);
    anomalyNode.rotation.x = Math.PI / 3;
    scene.add(anomalyNode);

    // Render loop
    function animate() {
        requestAnimationFrame(animate);

        anomalyNode.rotation.z += 0.006;
        grid.position.z = (Date.now() * 0.0015) % 1;

        const posArr = emberParticles.geometry.attributes.position.array;
        for (let i = 1; i < emberCount * 3; i += 3) {
            posArr[i] += 0.015;
            if (posArr[i] > 12) posArr[i] = -2;
        }
        emberParticles.geometry.attributes.position.needsUpdate = true;

        renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
</script>
</body>
</html>
"""

# Render Three.js Canvas
components.html(THREE_CANVAS_HTML, height=180, scrolling=False)

# -----------------------------------------------------------------------------
# NAVBAR
# -----------------------------------------------------------------------------
nav_col1, nav_col2 = st.columns([8, 4])
with nav_col1:
    st.markdown("<h3 style='margin:0; font-family:\"Instrument Serif\", serif; font-size: 2.2rem;'><span class='red-accent'>SMART ATTENDANCE</span> <span style='font-family:\"JetBrains Mono\", monospace; font-size:0.85rem; color:var(--text-muted);'>// ANOMALY DETECTOR ENGINE</span></h3>", unsafe_allow_html=True)
with nav_col2:
    st.markdown("<p style='font-family:\"JetBrains Mono\", monospace; color:var(--text-muted); font-size:0.8rem; margin-top:12px; text-align:right;'>SYSTEM STATUS: ONLINE | HYBRID ENSEMBLE ACTIVE</p>", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# HERO SECTION
# -----------------------------------------------------------------------------
hero_col1, hero_col2 = st.columns([8, 4], gap="large")

with hero_col1:
    st.markdown("""
        <div class="badge-tag">
            <span>●</span> UNSUPERVISED MACHINE LEARNING RUNTIME
        </div>
        <div class="hero-header">
            Smart Attendance <br>
            <span class="red-accent">Anomaly & Fraud Detector</span>
        </div>
        <p style="color: var(--text-muted); font-size: 1.05rem; margin-top: 14px; line-height: 1.6;">
            Detect proxy check-ins, arrival time deviations, and suspicious absence streaks using hybrid tree-neural ensemble intelligence.
        </p>
    """, unsafe_allow_html=True)

with hero_col2:
    st.markdown("""
    <div class="shelf-card">
        <div class="card-index">SYSTEM SPECIFICATIONS</div>
        <p style="color:#e2e8f0; font-family:'JetBrains Mono', monospace; font-size:0.8rem; line-height:1.8; margin:0;">
            • <b>Pipeline</b>: Isolation Forest + Autoencoder<br>
            • <b>Domain Rules</b>: Streaks & Arrival Deviations<br>
            • <b>Features</b>: Per-student baseline z-scores<br>
            • <b>Evaluation</b>: 30% held-out test split
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🎛️ SYSTEM CONTROLS")
st.sidebar.caption("// DATASET & HYPERPARAMETER CONFIG")

data_source = st.sidebar.radio("Data Source", ["Synthetic Dataset", "Upload Custom CSV"])

if data_source == "Synthetic Dataset":
    col_s1, col_s2 = st.sidebar.columns(2)
    n_students = col_s1.slider("Students", 20, 100, 60, step=10)
    n_days = col_s2.slider("Days", 30, 120, 90, step=15)
    anomaly_rate = st.sidebar.slider("Anomaly Rate Target (%)", 1, 15, 5) / 100.0

    @st.cache_data(show_spinner="Generating dataset...")
    def get_synth_data(n_stu, n_d, a_rate):
        return generate_attendance(n_students=n_stu, n_days=n_d, anomaly_rate=a_rate, seed=42)

    df_raw = get_synth_data(n_students, n_days, anomaly_rate)

else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV (Columns: student_id, date, check_in_minute, duration_minutes, present)",
        type=["csv"]
    )
    if uploaded_file is None:
        st.info("👈 Upload CSV file or switch to 'Synthetic Dataset'.")
        st.stop()
    df_raw = pd.read_csv(uploaded_file)
    if "is_anomaly" not in df_raw.columns:
        df_raw["is_anomaly"] = 0

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 DETECTOR SELECTION")

model_choice = st.sidebar.selectbox(
    "Active Detector",
    ["Hybrid Ensemble (ours)", "Isolation Forest", "Autoencoder"]
)

threshold_pct = st.sidebar.slider("Threshold Percentile", 85.0, 99.5, 95.0, step=0.5)

# -----------------------------------------------------------------------------
# CACHED MODEL EXECUTION ENGINE
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Fitting detector & calculating anomaly scores...")
def train_and_score(df_json: str, model_name: str, pct: float):
    df = pd.read_json(io.StringIO(df_json))
    df["date"] = df["date"].astype(str).str[:10]
    X = build_features(df)

    if model_name == "Isolation Forest":
        det = IsolationForestDetector()
        det.fit(X)
        scores = det.score(X)
        threshold = np.percentile(scores, pct)
        preds = (scores > threshold).astype(int)

    elif model_name == "Autoencoder":
        det = AutoencoderDetector(threshold_percentile=pct)
        det.fit(X)
        scores = det.score(X)
        preds = (scores > det.threshold_).astype(int)

    else:
        det = HybridDetector(
            [IsolationForestDetector(), AutoencoderDetector()],
            threshold_percentile=pct
        )
        det.fit(X)
        scores = det.score(X)
        preds = (scores > det.threshold_).astype(int)

    return X, scores, preds

X_mat, raw_scores, predictions = train_and_score(df_raw.to_json(), model_choice, threshold_pct)

df_processed = categorize_and_explain(df_raw, X_mat, raw_scores, predictions)
df_processed["anomaly_score"] = np.round(raw_scores, 4)
df_processed["flagged"] = predictions

# -----------------------------------------------------------------------------
# TABS NAVIGATION
# -----------------------------------------------------------------------------
tab_overview, tab_analytics, tab_student, tab_benchmarks = st.tabs([
    "🛰️ LIVE INTELLIGENCE",
    "📊 DEEP ANALYTICS",
    "🕵️ STUDENT INSPECTOR",
    "⚖️ MODEL BENCHMARKS"
])

# =============================================================================
# TAB 1: LIVE INTELLIGENCE
# =============================================================================
with tab_overview:
    total_records = len(df_processed)
    total_flagged = int(df_processed["flagged"].sum())
    true_anomalies = int(df_processed["is_anomaly"].sum())
    critical_count = len(df_processed[df_processed["risk_severity"] == "Critical"])

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="shelf-card">
            <div class="card-index">OVERVIEW // 01</div>
            <div style="color:var(--text-muted); font-size:0.8rem;">TOTAL RECORDS</div>
            <div class="metric-value-display">{total_records:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="shelf-card">
            <div class="card-index">OVERVIEW // 02</div>
            <div style="color:var(--text-muted); font-size:0.8rem;">FLAGGED ANOMALIES</div>
            <div class="metric-value-display" style="color:var(--accent-red);">{total_flagged:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        if true_anomalies > 0:
            tp = int(((df_processed["flagged"] == 1) & (df_processed["is_anomaly"] == 1)).sum())
            recall_val = round((tp / true_anomalies) * 100, 1)
            st.markdown(f"""
            <div class="shelf-card">
                <div class="card-index">OVERVIEW // 03</div>
                <div style="color:var(--text-muted); font-size:0.8rem;">GROUND RECALL</div>
                <div class="metric-value-display" style="color:#10B981;">{tp}/{true_anomalies} ({recall_val}%)</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="shelf-card">
                <div class="card-index">OVERVIEW // 03</div>
                <div style="color:var(--text-muted); font-size:0.8rem;">FLAGGED RATE</div>
                <div class="metric-value-display" style="color:#A855F7;">{round((total_flagged/total_records)*100, 1)}%</div>
            </div>
            """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="shelf-card">
            <div class="card-index">OVERVIEW // 04</div>
            <div style="color:var(--text-muted); font-size:0.8rem;">CRITICAL SEVERITY</div>
            <div class="metric-value-display" style="color:var(--accent-red);">{critical_count:,}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("<h5 style='font-family:\"JetBrains Mono\", monospace; color:var(--accent-red);'>TAXONOMY DISTRIBUTION</h5>", unsafe_allow_html=True)
        cat_counts = df_processed[df_processed["flagged"] == 1]["anomaly_category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]

        if not cat_counts.empty:
            fig_pie = px.pie(
                cat_counts, values="Count", names="Category",
                color_discrete_sequence=["#e0231c", "#00F2FE", "#A855F7", "#FBBF24", "#3B82F6"],
                hole=0.5
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#8a919e",
                margin=dict(l=10, r=10, t=20, b=20),
                showlegend=True
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No anomalies flagged at current threshold.")

    with col_right:
        st.markdown("<h5 style='font-family:\"JetBrains Mono\", monospace; color:var(--accent-red);'>DETECTED SUSPICIOUS RECORDS</h5>", unsafe_allow_html=True)
        
        severity_filter = st.multiselect(
            "Filter Severity",
            ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High", "Medium"]
        )
        
        filtered_df = df_processed[
            (df_processed["flagged"] == 1) & 
            (df_processed["risk_severity"].isin(severity_filter))
        ].sort_values("anomaly_score", ascending=False)

        st.dataframe(
            filtered_df[[
                "student_id", "date", "check_in_minute", "duration_minutes", 
                "risk_severity", "anomaly_category", "anomaly_reason", "anomaly_score"
            ]],
            column_config={
                "student_id": "Student ID",
                "date": "Date",
                "check_in_minute": st.column_config.NumberColumn("Arrival (Min)", format="%.0f min"),
                "duration_minutes": st.column_config.NumberColumn("Duration (Min)", format="%.0f min"),
                "risk_severity": "Severity",
                "anomaly_category": "Category",
                "anomaly_reason": "Reason",
                "anomaly_score": st.column_config.ProgressColumn("Suspicion Score", min_value=0.0, max_value=1.0, format="%.3f")
            },
            use_container_width=True,
            height=340
        )

        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 EXPORT ANOMALY AUDIT REPORT (CSV)",
            data=csv_data,
            file_name=f"anomalies_{model_choice.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )

# =============================================================================
# TAB 2: DEEP ANALYTICS
# =============================================================================
with tab_analytics:
    st.markdown("<h4 style='font-family:\"Instrument Serif\", serif;'>Feature Space & Score Distributions</h4>", unsafe_allow_html=True)

    a_col1, a_col2 = st.columns(2)

    with a_col1:
        st.markdown("<h5 style='font-family:\"JetBrains Mono\", monospace; color:var(--text-muted);'>ARRIVAL VS STAY DURATION SCATTER</h5>", unsafe_allow_html=True)
        
        fig_scatter = px.scatter(
            df_processed,
            x="check_in_minute",
            y="duration_minutes",
            color=df_processed["flagged"].map({1: "Anomaly Flagged", 0: "Normal Attendance"}),
            color_discrete_map={"Normal Attendance": "#1f232d", "Anomaly Flagged": "#e0231c"},
            hover_data=["student_id", "date", "anomaly_category", "anomaly_score"],
            labels={"check_in_minute": "Check-in Minute", "duration_minutes": "Stay Duration (Min)"}
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18, 20, 26, 0.8)",
            font_color="#8a919e",
            legend_title_text="",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with a_col2:
        st.markdown("<h5 style='font-family:\"JetBrains Mono\", monospace; color:var(--text-muted);'>ANOMALY SCORE HISTOGRAM</h5>", unsafe_allow_html=True)

        fig_hist = px.histogram(
            df_processed,
            x="anomaly_score",
            color=df_processed["flagged"].map({1: "Flagged", 0: "Normal"}),
            color_discrete_map={"Normal": "#00F2FE", "Flagged": "#e0231c"},
            nbins=40,
            labels={"anomaly_score": "Score"}
        )
        cutoff_val = np.percentile(raw_scores, threshold_pct)
        fig_hist.add_vline(x=cutoff_val, line_dash="dash", line_color="#FBBF24", annotation_text=f"Cutoff ({threshold_pct}%)")
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18, 20, 26, 0.8)",
            font_color="#8a919e",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    st.markdown("<h5 style='font-family:\"JetBrains Mono\", monospace; color:var(--text-muted);'>DAILY ANOMALY INCIDENCE TIMELINE</h5>", unsafe_allow_html=True)
    
    timeline_df = df_processed.groupby(["date", "flagged"]).size().unstack(fill_value=0).reset_index()
    if 1 in timeline_df.columns:
        fig_timeline = px.line(
            timeline_df,
            x="date",
            y=1,
            labels={"date": "Date", "1": "Anomalies Flagged"},
            line_shape="spline"
        )
        fig_timeline.update_traces(line_color="#e0231c", line_width=2.5)
        fig_timeline.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18, 20, 26, 0.8)",
            font_color="#8a919e",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

# =============================================================================
# TAB 3: STUDENT INSPECTOR
# =============================================================================
with tab_student:
    st.markdown("<h4 style='font-family:\"Instrument Serif\", serif;'>Student Profile & Historical Audit</h4>", unsafe_allow_html=True)

    all_students = sorted(df_processed["student_id"].unique())
    selected_student = st.selectbox("Select Student ID", all_students)

    stu_df = df_processed[df_processed["student_id"] == selected_student].sort_values("date")

    s_present_cnt = int(stu_df["present"].sum())
    s_total_days = len(stu_df)
    s_att_pct = round((s_present_cnt / s_total_days) * 100, 1)
    s_anomalies_cnt = int(stu_df["flagged"].sum())
    s_avg_checkin = stu_df[stu_df["present"] == 1]["check_in_minute"].median()
    s_avg_dur = stu_df[stu_df["present"] == 1]["duration_minutes"].median()

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Overall Attendance", f"{s_att_pct}%", f"{s_present_cnt}/{s_total_days} Days")
    sc2.metric("Flagged Anomalies", f"{s_anomalies_cnt}", delta_color="inverse")
    sc3.metric("Habitual Arrival", f"{round(s_avg_checkin/60, 2)} hrs", f"{round(s_avg_checkin)} min past 00:00")
    sc4.metric("Habitual Duration", f"{round(s_avg_dur, 1)} min")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"<h5 style='font-family:\"JetBrains Mono\", monospace; color:var(--accent-red);'>90-DAY ATTENDANCE TIMELINE FOR STUDENT #{selected_student}</h5>", unsafe_allow_html=True)
    
    fig_stu = go.Figure()
    fig_stu.add_trace(go.Scatter(
        x=stu_df["date"],
        y=stu_df["check_in_minute"],
        mode="lines+markers",
        name="Arrival Minute",
        line=dict(color="#00F2FE", width=1.5),
        marker=dict(size=6)
    ))
    
    anom_points = stu_df[stu_df["flagged"] == 1]
    if not anom_points.empty:
        fig_stu.add_trace(go.Scatter(
            x=anom_points["date"],
            y=anom_points["check_in_minute"],
            mode="markers",
            name="Anomaly Flagged",
            marker=dict(color="#e0231c", size=12, symbol="triangle-up")
        ))

    fig_stu.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(18, 20, 26, 0.8)",
        font_color="#8a919e",
        xaxis_title="Date",
        yaxis_title="Arrival Minute",
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_stu, use_container_width=True)

    st.dataframe(
        stu_df[[
            "date", "check_in_minute", "duration_minutes", "present",
            "anomaly_category", "anomaly_reason", "anomaly_score", "flagged"
        ]],
        use_container_width=True
    )

# =============================================================================
# TAB 4: BENCHMARKS
# =============================================================================
with tab_benchmarks:
    st.markdown("<h4 style='font-family:\"Instrument Serif\", serif;'>Quantitative Model Comparison & Matrix</h4>", unsafe_allow_html=True)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve

    X_full = build_features(df_raw)
    y_full = df_raw["is_anomaly"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y_full, test_size=0.3, stratify=y_full, random_state=42
    )

    models = {
        "Isolation Forest": IsolationForestDetector(),
        "Autoencoder": AutoencoderDetector(),
        "Hybrid Ensemble (ours)": HybridDetector([IsolationForestDetector(), AutoencoderDetector()])
    }

    benchmark_results = []
    roc_data = {}

    for name, det in models.items():
        det.fit(X_tr)
        preds = det.predict(X_te)
        scs = det.score(X_te)

        prec = precision_score(y_te, preds, zero_division=0)
        rec = recall_score(y_te, preds, zero_division=0)
        f1 = f1_score(y_te, preds, zero_division=0)
        auc = roc_auc_score(y_te, scs)

        fpr, tpr, _ = roc_curve(y_te, scs)
        roc_data[name] = (fpr, tpr)

        benchmark_results.append({
            "Detector Model": name,
            "Precision": round(prec, 3),
            "Recall": round(rec, 3),
            "F1-Score": round(f1, 3),
            "ROC-AUC": round(auc, 3)
        })

    bench_df = pd.DataFrame(benchmark_results)

    b_col1, b_col2 = st.columns([1, 1])

    with b_col1:
        st.markdown("<h5 style='font-family:\"JetBrains Mono\", monospace; color:var(--accent-red);'>TEST SET METRICS (30% SPLIT)</h5>", unsafe_allow_html=True)
        st.dataframe(bench_df, use_container_width=True, height=180)

        fig_bar = px.bar(
            bench_df.melt(id_vars="Detector Model", var_name="Metric", value_name="Value"),
            x="Detector Model",
            y="Value",
            color="Metric",
            barmode="group",
            color_discrete_sequence=["#e0231c", "#00F2FE", "#A855F7", "#10B981"]
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18, 20, 26, 0.8)",
            font_color="#8a919e",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with b_col2:
        st.markdown("<h5 style='font-family:\"JetBrains Mono\", monospace; color:var(--accent-red);'>ROC CURVES</h5>", unsafe_allow_html=True)

        fig_roc = go.Figure()
        colors = {"Isolation Forest": "#00F2FE", "Autoencoder": "#A855F7", "Hybrid Ensemble (ours)": "#e0231c"}

        for name, (fpr, tpr) in roc_data.items():
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=name, line=dict(color=colors[name], width=2)
            ))

        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random baseline", line=dict(color="#8a919e", dash="dash")
        ))

        fig_roc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18, 20, 26, 0.8)",
            font_color="#8a919e",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
f_col1, f_col2 = st.columns([6, 6])
with f_col1:
    st.markdown("<p style='font-family:\"JetBrains Mono\", monospace; color:var(--text-muted); font-size:0.8rem;'>SMART ATTENDANCE ANOMALY DETECTOR SYSTEM</p>", unsafe_allow_html=True)
with f_col2:
    st.markdown("<p style='font-family:\"JetBrains Mono\", monospace; color:var(--text-muted); font-size:0.8rem; text-align:right;'>THREE.JS • STREAMLIT • PYDATA STACK</p>", unsafe_allow_html=True)
