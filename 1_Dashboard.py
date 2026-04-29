import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Telco Churn Analyzer Dashboard", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    /* Premium Dark Accent Metric Cards (Matched to Page 2) */
    div[data-testid="stMetric"] {
        background-color: #1A2235 !important; /* Deep dark slate */
        border: 1px solid #2D3243 !important;
        border-left: 5px solid #4da6ff !important; /* Sharp blue accent line */
        padding: 16px 20px !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.2s ease-in-out !important; /* Keeps the smooth hover animation */
    }

    /* The Hover Effect */
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.2) !important;
        transform: translateY(-3px) !important; 
    }

    /* Sharp, crisp label text */
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] label p {
        color: #8b949e !important; 
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important; 
        letter-spacing: 1px !important;
    }

    /* Giant, bright white value text */
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { 
        font-size: 2.2rem !important; 
        font-weight: 800 !important; 
        color: #ffffff !important; 
    }

    div[data-testid="stMetric"] label { font-size: 0.85rem; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.6rem; }

    /* Expander styling */
    .streamlit-expanderHeader { font-weight: 600; font-size: 0.95rem; }

    /* Table tweaks */
    thead th { font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.5px; }

    /* Hide default Streamlit footer */
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
            
    /* Make the Delta indicator (like "-1,869 lost") a Sharp Bright Green */
    div[data-testid="stMetricDelta"] > div {
        color: #2EEB79 !important; /* Sharp, bright mint green */
        font-weight: 700 !important; /* Makes it slightly bolder */
    }
    
    div[data-testid="stMetricDelta"] svg {
        fill: #2EEB79 !important; /* Makes the little arrow match the text */
    }
</style>
""", unsafe_allow_html=True)

st.markdown("## Dashboard")
st.caption("Key performance indicators, model summary, and top churn drivers")

# Historical Baseline Disclaimer
st.markdown("""
<div style="background-color: rgba(77, 166, 255, 0.1); padding: 10px; border-radius: 5px; border-left: 4px solid #4da6ff; margin-bottom: 20px; margin-top: 10px;">
    <strong> Historical Baseline Analysis</strong><br>
    <span style="font-size: 14px; color: inherit;">The insights on this page are derived from the foundational <b>IBM Telco Dataset (7,043 historical records)</b>. They serve as the baseline for our predictive models.</span>
</div>
""", unsafe_allow_html=True)
st.divider()

# KPI Cards
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Customers", "7,043", help="IBM Telco dataset")
k2.metric("Churn Rate", "26.5%", delta="-1,869 lost", delta_color="inverse")
k3.metric("Avg Tenure", "32.4 mo", help="Range: 0 – 72 months")
k4.metric("Avg Monthly Charges", "$64.76", help="Range: $18.25 – $118.75")

st.divider()

# Cluster Pie + Model Comparison
col_pie, col_table = st.columns([1, 1.2])

with col_pie:
    st.markdown("#### Customer Segments (K-Means, k=3)")

    fig_pie = go.Figure(go.Pie(
        labels=["Cluster 0: High Churn", "Cluster 1: Loyal Premium", "Cluster 2: Basic Offline"],
        values=[2732, 2675, 1636],
        hole=0.5,
        marker=dict(
            colors=["#ff4b4b", "#4da6ff", "#3fb950"],
            line=dict(color="rgba(0,0,0,0)", width=2)
        ),
        textinfo="percent+label",
        textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>Customers: %{value:,}<br>Share: %{percent}<extra></extra>"
    ))
    fig_pie.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_table:
    st.markdown("#### Tuned Model Comparison (Threshold = 0.40)")

    df_models = pd.DataFrame({
        "Metric": ["Accuracy", "Churn Precision", "Churn Recall", "Churn F1-Score", "ROC-AUC"],
        "RF ★": [0.75, 0.52, 0.81, 0.63, 0.85],
        "SVM":   [0.75, 0.52, 0.76, 0.62, 0.82],
        "LR":    [0.75, 0.52, 0.78, 0.62, 0.85],
        "XGBoost": [0.76, 0.53, 0.77, 0.63, 0.85],
    })
    df_models = df_models.set_index("Metric")

    def highlight_rf(col):
        """Highlight RF column and best Recall value."""
        styles = []
        for i, val in enumerate(col):
            if col.name == "RF ★":
                if i == 2:  # Recall row
                    styles.append("background-color: rgba(255,75,75,0.2); color: #ff6b6b; font-weight: 700;")
                else:
                    styles.append("background-color: rgba(255,75,75,0.05);")
            else:
                styles.append("")
        return styles

    st.dataframe(
        df_models.style.apply(highlight_rf).format("{:.2f}"),
        use_container_width=True,
        height=230
    )

    st.success(
        "**★ Best Model: Random Forest** — catches **81%** of churners (Recall), "
        "3-5% more than other models. In telco, missing a churner costs far more "
        "than a false positive."
    )

st.divider()

# SHAP Global Feature Importance
st.markdown("#### Top Churn Drivers — SHAP Global Importance")

shap_data = pd.DataFrame({
    "Feature": [    
        "Internet Service: Fiber optic",
        "Contract: Two year",
        "Tenure Months",
        "Dependents: Yes",
        "Contract: One year",
        "Payment: Electronic check",
    ],
    "SHAP Value": [0.068, 0.067, 0.065, 0.065, 0.042, 0.038],
    "Direction": [
        "↑ Increases churn",
        "↓ Decreases churn",
        "↓ Longer = loyal",
        "↓ More stable",
        "↓ Commitment helps",
        "↑ Increases churn",
    ],
    "Color": ["#ff4b4b", "#4da6ff", "#4da6ff", "#4da6ff", "#4da6ff", "#ff4b4b"],
})

fig_shap = go.Figure()
fig_shap.add_trace(go.Bar(
    y=shap_data["Feature"],
    x=shap_data["SHAP Value"],
    orientation="h",
    marker_color=shap_data["Color"],
    text=shap_data.apply(lambda r: f'{r["SHAP Value"]:.3f}  {r["Direction"]}', axis=1),
    textposition="outside",
    textfont=dict(size=12),
    hovertemplate="<b>%{y}</b><br>SHAP: %{x:.3f}<extra></extra>"
))
fig_shap.update_layout(
    height=280,
    margin=dict(t=10, b=10, l=10, r=200),
    xaxis=dict(title="Mean |SHAP Value|", showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)"), # Softened gridline color
    yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_shap, use_container_width=True)

st.caption(
    "Contract type, internet service, tenure, and household status dominate the model's "
    "prediction logic. Fiber optic users and electronic check payers show highest churn tendency."
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

# Landing page (when no sub-page is selected)
st.info("👈 Pick a page from the sidebar to explore.")
