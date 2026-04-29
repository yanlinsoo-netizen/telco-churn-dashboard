import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(page_title="Segmentation Explorer", layout="wide")

# Inject CSS
st.markdown("""
<style>
    /* 1. Make table text slightly smaller and force it to wrap */
    .stDataFrame td, .stDataFrame th {
        font-size: 13px !important;
        white-space: normal !important; 
    }
    
    /* 2. Premium Dark Accent Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1A2235 !important; /* Deep dark slate */
        border: 1px solid #2D3243 !important;
        border-left: 5px solid #4da6ff !important; /* Sharp blue accent line */
        padding: 15px 20px !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15) !important;
        transition: all 0.2s ease-in-out !important;
    }

    div[data-testid="stMetric"]:hover {
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.2) !important;
        transform: translateY(-3px) !important;
    }
    
    /* 3. Stop Metric Values from truncating and make them bright white */
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        white-space: normal !important; 
        word-break: break-word !important; 
        font-size: 1.2rem !important; 
        font-weight: 800 !important;
        color: #ffffff !important;
    }
    
    /* 4. Crisp Metric Labels */
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] label p {
        white-space: normal !important;
        color: #8b949e !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
</style>
""", unsafe_allow_html=True)

# Segmentation
st.markdown("## Segmentation Explorer")
st.caption("Interactive K-Means cluster profiles, evaluation metrics, and feature comparison")

# Historical Baseline Disclaimer
st.markdown("""
<div style="background-color: rgba(77, 166, 255, 0.1); padding: 10px; border-radius: 5px; border-left: 4px solid #4da6ff; margin-bottom: 20px; margin-top: 10px;">
    <strong> Historical Baseline Analysis</strong><br>
    <span style="font-size: 14px; color: inherit;">The insights on this page are derived from the foundational <b>IBM Telco Dataset (7,043 historical records)</b>. They serve as the baseline for our predictive models.</span>
</div>
""", unsafe_allow_html=True)

st.divider()

# Clustering Evaluation Metrics
st.markdown("#### Clustering Evaluation Metrics (K-Means, k=3)")

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Silhouette Score", "0.321")
    st.caption("Higher = better separation (range -1 to 1)")
with m2:
    st.metric("Davies-Bouldin Index", "1.25")
    st.caption("Lower = better (0 = perfect)")
with m3:
    st.metric("Calinski-Harabasz Index", "3,742.55")
    st.caption("Higher = denser, well-separated clusters")

st.divider()

# 3 Cluster Profile Cards
st.markdown("#### Customer Segment Profiles")

c0, c1, c2 = st.columns(3)

with c0:
    st.markdown("""
    <div style="border:1px solid rgba(255,75,75,0.3); border-radius:12px; padding:18px; 
                background:rgba(255,75,75,0.05);">
        <span style="font-family:monospace; font-size:11px; color:#545d6a;">CLUSTER 0</span><br>
        <span style="font-size:18px; font-weight:700; color:#ff6b6b;">High Churn Customers</span><br>
        <span style="background:rgba(255,75,75,0.15); color:#ff6b6b; font-size:11px; 
              font-weight:600; padding:2px 8px; border-radius:10px;">⚠️ HIGH RISK</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Avg Tenure", "15.3 mo")
    col_b.metric("Monthly", "$67.95")
    col_c.metric("Total", "$1,018")
    st.markdown("""
    | Feature | Value |
    |:--------|------:|
    | Contract M2M | **85%** |
    | Internet | 100% |
    | Online Security | 24% |
    | Tech Support | 24% |
    | Phone Service | 84% |
    """)

with c1:
    st.markdown("""
    <div style="border:1px solid rgba(77,166,255,0.3); border-radius:12px; padding:18px; 
                background:rgba(77,166,255,0.05);">
        <span style="font-family:monospace; font-size:11px; color:#545d6a;">CLUSTER 1</span><br>
        <span style="font-size:18px; font-weight:700; color:#4da6ff;">Loyal Premium</span><br>
        <span style="background:rgba(63,185,80,0.15); color:#3fb950; font-size:11px; 
              font-weight:600; padding:2px 8px; border-radius:10px;">✅ LOW RISK</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Avg Tenure", "57.8 mo")
    col_b.metric("Monthly", "$89.51")
    col_c.metric("Total", "$5,158")
    st.markdown("""
    | Feature | Value |
    |:--------|------:|
    | Long-term Contract | **73%** |
    | Internet | 100% |
    | Online Security | **54%** |
    | Tech Support | **56%** |
    | Auto Payment | 63% |
    """)

with c2:
    st.markdown("""
    <div style="border:1px solid rgba(63,185,80,0.3); border-radius:12px; padding:18px; 
                background:rgba(63,185,80,0.05);">
        <span style="font-family:monospace; font-size:11px; color:#545d6a;">CLUSTER 2</span><br>
        <span style="font-size:18px; font-weight:700; color:#3fb950;">Basic Offline</span><br>
        <span style="background:rgba(63,185,80,0.15); color:#3fb950; font-size:11px; 
              font-weight:600; padding:2px 8px; border-radius:10px;">✅ LOW RISK</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Avg Tenure", "30.6 mo")
    col_b.metric("Monthly", "$21.08")
    col_c.metric("Total", "$663")
    st.markdown("""
    | Feature | Value |
    |:--------|------:|
    | Two-year Contract | 42% |
    | Internet | **0%** |
    | Online Security | 0% |
    | Mailed Check | 49% |
    | Phone Service | **100%** |
    """)

st.divider()

# Radar Chart + Detailed Feature Table
col_radar, col_detail = st.columns([1, 1])

with col_radar:
    st.markdown("#### Cluster Feature Comparison (Radar)")

    categories = ["Tenure", "Monthly $", "Total $", "Services", "Long Contract", "Auto Pay"]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=[0.21, 0.57, 0.12, 0.29, 0.15, 0.34, 0.21],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(255,75,75,0.08)",
        line=dict(color="#ff4b4b", width=2),
        name="C0: High Churn"
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=[0.80, 0.75, 0.59, 0.65, 0.73, 0.63, 0.80],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(77,166,255,0.08)",
        line=dict(color="#4da6ff", width=2),
        name="C1: Loyal Premium"
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=[0.42, 0.18, 0.08, 0.00, 0.42, 0.22, 0.42],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(63,185,80,0.08)",
        line=dict(color="#3fb950", width=2),
        name="C2: Basic Offline"
    ))
    
    # Update plotly layout for light theme
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False,
                            gridcolor="rgba(128,128,128,0.2)"),
            angularaxis=dict(gridcolor="rgba(128,128,128,0.2)")
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5,
                   font=dict(size=11)),
        height=400,
        margin=dict(t=30, b=60, l=85, r=85), 
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1e2333", size=10), 
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with col_detail:
    st.markdown("#### Detailed Feature Comparison")

    df_features = pd.DataFrame({
        "Feature": [
            "Tenure Months", "Monthly Charges", "Total Charges",
            "Phone Service", "Multiple Lines", "Internet Service",
            "Online Security", "Online Backup", "Device Protection",
            "Tech Support", "Streaming TV", "Streaming Movies",
            "Contract (M2M)", "Paperless Billing"
        ],
        "C0: High Churn": [
            "15.34", "$67.95", "$1,018",
            "84%", "32%", "100%",
            "24%", "28%", "27%",
            "24%", "33%", "34%",
            "85%", "66%"
        ],
        "C1: Loyal": [
            "57.82", "$89.51", "$5,158",
            "93%", "70%", "100%",
            "54%", "67%", "69%",
            "56%", "71%", "72%",
            "26%", "69%"
        ],
        "C2: Basic": [
            "30.55", "$21.08", "$663",
            "100%", "22%", "0%",
            "0%", "0%", "0%",
            "0%", "0%", "0%",
            "34%", "29%"
        ],
    })
    df_features = df_features.set_index("Feature")

    def color_extremes(val):
        """Highlight notable values."""
        try:
            v = float(val.replace("$", "").replace(",", "").replace("%", ""))
            if "%" in str(val):
                if v >= 85:
                    return "color: #ff6b6b; font-weight: 600;"
                elif v == 0:
                    return "color: #ff6b6b; font-weight: 600;"
                elif v >= 70:
                    return "color: #3fb950; font-weight: 600;"
        except:
            pass
        return ""

    st.dataframe(
        df_features.style.map(color_extremes),
        use_container_width=True,
        height=520
    )

st.divider()

# Clustering Algorithm Comparison
st.markdown("#### Clustering Algorithm Comparison")

df_cluster_eval = pd.DataFrame({
    "Model": ["K-Means ★", "K-Medoids", "Random Clustering"],
    "Silhouette ↑": [0.321, 0.314, -0.003],
    "DBI ↓": [1.25, 1.28, 82.58],
    "CHI ↑": [3742.55, 3666.22, 0.75],
    "Interpretation": [
        "Best — well-separated clusters",
        "Good — similar to K-Means",
        "Baseline — no meaningful structure"
    ]
})
df_cluster_eval = df_cluster_eval.set_index("Model")

def highlight_best_model(row):
    if row.name == "K-Means ★":
        return ["background-color: rgba(255,75,75,0.08); font-weight: 600;"] * len(row)
    return [""] * len(row)

st.dataframe(
    df_cluster_eval.style.apply(highlight_best_model, axis=1),
    use_container_width=True,
)

st.caption(
    "K-Means outperforms K-Medoids across all 3 metrics. "
    "Random Clustering (negative Silhouette) confirms that the K-Means clusters "
    "capture real data structure, not random noise."
)

# Sidebar
with st.sidebar:
    st.markdown("# Telco Churn Analyzer")
    st.markdown("""
    **Best Model:** Random Forest  
    **Threshold:** 0.40  
    **Dataset:** IBM Telco · 7043 rows
    """)
    st.divider()
    st.caption("© WQD7012 Group 6 — AML Project")