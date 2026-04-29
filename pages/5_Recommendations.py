import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Recommendations", layout="wide")

# Inject CSS
st.markdown("""
<style>
    /* Make the Strategy Cards "Float" with Hover Effects */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #4da6ff !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08) !important;
        transform: translateY(-3px) !important;
    }
</style>
""", unsafe_allow_html=True)

# Retention Recommendations
st.markdown("## Retention Recommendations")
st.caption("SHAP-driven, cluster-specific retention strategies based on model insights")

# Historical Baseline Disclaimer
st.markdown("""
<div style="background-color: rgba(77, 166, 255, 0.1); padding: 10px; border-radius: 5px; border-left: 4px solid #4da6ff; margin-bottom: 20px; margin-top: 10px;">
    <strong> Historical Baseline Analysis</strong><br>
    <span style="font-size: 14px; color: inherit;">The insights on this page are derived from the foundational <b>IBM Telco Dataset (7,043 historical records)</b>. They serve as the baseline for our predictive models.</span>
</div>
""", unsafe_allow_html=True)
st.divider()

# Cluster Selector
cluster_choice = st.radio(
    "Select customer segment to view strategies:",
    ["All Clusters", "Cluster 0: High Churn (Critical)", "Cluster 1: Loyal Premium (Best)", "Cluster 2: Basic Offline (Stable)"],
    horizontal=True,
)

st.divider()

# CLUSTER 0: HIGH CHURN
if cluster_choice in ["All Clusters", "Cluster 0: High Churn (Critical)"]:
    st.markdown("""
    <div style="border:1px solid rgba(255,75,75,0.2); border-radius:12px; padding:14px 18px; 
                background:rgba(255,75,75,0.05); margin-bottom:16px;">
        <span style="font-size:16px; font-weight:700; color:#ff6b6b;">
            ⚠️ Cluster 0 — High Churn Customers (Critical)
        </span>
        <span style="float:right; background:rgba(255,75,75,0.15); color:#ff6b6b; 
              font-size:11px; font-weight:600; padding:3px 10px; border-radius:10px;">
            URGENT ACTION NEEDED
        </span>
    </div>
    """, unsafe_allow_html=True)

    r0_1, r0_2 = st.columns(2)

    with r0_1:
        with st.container(border=True):
            st.markdown("##### Contract Migration Incentive")
            st.markdown(
                "85% are on **month-to-month contracts** — the #1 churn driver. "
                "Offer **15-20% discount** for switching to 12-month or 24-month plans."
            )
            st.caption("SHAP evidence: `Contract_Two year` has the 2nd highest protective effect (0.067)")
            st.markdown("`Impact: Very High` · `Effort: Low` · `SHAP: Contract`")

        with st.container(border=True):
            st.markdown("##### Service Bundle Upgrade")
            st.markdown(
                "Only **24%** have Online Security and Tech Support — SHAP confirms "
                "these services decrease churn. Offer **free 3-month trial** of "
                "security + tech support bundle to increase engagement."
            )
            st.caption("SHAP evidence: `Online Security` and `Tech Support` both reduce churn")
            st.markdown("`Impact: High` · `Effort: Medium` · `SHAP: Online Security, Tech Support`")

    with r0_2:
        with st.container(border=True):
            st.markdown("##### Payment Method Migration")
            st.markdown(
                "Electronic check users show elevated churn (SHAP +0.038). "
                "Incentivize switch to **auto-pay** (bank transfer / credit card) "
                "with a **one-time $20 credit**. Reduces payment friction."
            )
            st.caption("SHAP evidence: `Payment_Electronic check` → +0.038 churn increase")
            st.markdown("`Impact: High` · `Effort: Low` · `SHAP: Payment Method`")

        with st.container(border=True):
            st.markdown("##### Early Tenure Intervention")
            st.markdown(
                "Average tenure is only **15.3 months** — Tenure is the 3rd strongest "
                "SHAP feature (0.065). Flag all customers with **tenure < 6 months** "
                "for proactive retention calls within the first 30 days."
            )
            st.caption("SHAP evidence: `Tenure Months` → 0.065 (longer tenure = less churn)")
            st.markdown("`Impact: Critical` · `Effort: Medium` · `SHAP: Tenure`")

    st.divider()

# CLUSTER 1: LOYAL PREMIUM
if cluster_choice in ["All Clusters", "Cluster 1: Loyal Premium (Best)"]:
    st.markdown("""
    <div style="border:1px solid rgba(77,166,255,0.2); border-radius:12px; padding:14px 18px; 
                background:rgba(77,166,255,0.05); margin-bottom:16px;">
        <span style="font-size:16px; font-weight:700; color:#4da6ff;">
            👑 Cluster 1 — Loyal Premium Customers (Best)
        </span>
        <span style="float:right; background:rgba(63,185,80,0.15); color:#3fb950; 
              font-size:11px; font-weight:600; padding:3px 10px; border-radius:10px;">
            MAINTAIN & GROW
        </span>
    </div>
    """, unsafe_allow_html=True)

    r1_1, r1_2, r1_3 = st.columns(3)

    with r1_1:
        with st.container(border=True):
            st.markdown("##### VIP Loyalty Program")
            st.markdown(
                "Most valuable customers (avg tenure **57.8 months**, **$89.51/mo**). "
                "Launch a tiered loyalty program: birthday discounts, anniversary "
                "rewards, exclusive early access to new services."
            )
            st.markdown("`Impact: Medium` · `Effort: Medium`")

    with r1_2:
        with st.container(border=True):
            st.markdown("##### Cross-sell Premium Tiers")
            st.markdown(
                "Already highly engaged (70% backup, 69% device protection). "
                "Offer **premium speed upgrades**, family plans, or bundled "
                "streaming packages to increase ARPU."
            )
            st.markdown("`Impact: High` · `Effort: Low`")

    with r1_3:
        with st.container(border=True):
            st.markdown("##### Referral Incentive")
            st.markdown(
                "Happy long-term customers are the best advocates. "
                "Offer **$25 credit** for each successful referral — leverages "
                "their satisfaction to acquire new customers at lower CAC."
            )
            st.markdown("`Impact: Medium` · `Effort: Low`")

    st.divider()

# CLUSTER 2: BASIC OFFLINE
if cluster_choice in ["All Clusters", "Cluster 2: Basic Offline (Stable)"]:
    st.markdown("""
    <div style="border:1px solid rgba(63,185,80,0.2); border-radius:12px; padding:14px 18px; 
                background:rgba(63,185,80,0.05); margin-bottom:16px;">
        <span style="font-size:16px; font-weight:700; color:#3fb950;">
            🌿 Cluster 2 — Basic Offline Customers (Stable)
        </span>
        <span style="float:right; background:rgba(63,185,80,0.15); color:#3fb950; 
              font-size:11px; font-weight:600; padding:3px 10px; border-radius:10px;">
            GROW OPPORTUNITY
        </span>
    </div>
    """, unsafe_allow_html=True)

    r2_1, r2_2, r2_3 = st.columns(3)

    with r2_1:
        with st.container(border=True):
            st.markdown("##### Digital Onboarding Campaign")
            st.markdown(
                "100% have no internet service — massive upsell opportunity. "
                "Offer entry-level internet bundle at **\$15/mo introductory rate** "
                "for 6 months. Can double ARPU from \$21 to \$40+."
            )
            st.markdown("`Impact: Very High` · `Effort: High`")

    with r2_2:
        with st.container(border=True):
            st.markdown("##### Payment Modernization")
            st.markdown(
                "**49%** pay by mailed check — highest friction method. "
                "Guide migration to online auto-pay with a small **monthly "
                "discount ($2 off)**. Reduces processing costs."
            )
            st.markdown("`Impact: Medium` · `Effort: Low`")

    with r2_3:
        with st.container(border=True):
            st.markdown("##### Maintain Stability")
            st.markdown(
                "42% already on two-year contracts with low churn risk. "
                "**Do not over-contact** — this segment values simplicity. "
                "Focus upsell on opt-in channels (mailer inserts, billing page)."
            )
            st.markdown("`Impact: Low` · `Effort: Low`")

    st.divider()

# Priority Matrix
st.markdown("#### Strategy Priority Matrix")
st.caption("Mapping all strategies by **expected impact** vs **implementation difficulty**")

fig_matrix = go.Figure()

# Quick Wins (High Impact, Easy)
fig_matrix.add_trace(go.Scatter(
    x=[1.5, 2, 1], y=[4.5, 4, 3.8],
    mode="markers+text",
    marker=dict(size=20, color="#39b14b", opacity=0.7),
    text=["Contract migration<br>discount", "Payment auto-pay<br>incentive", "Early tenure<br>retention call"],
    textposition="middle right", textfont=dict(size=15, color="#1e2333"),
    name="Quick Wins", hoverinfo="text"
))

# Strategic Bets (High Impact, Hard)
fig_matrix.add_trace(go.Scatter(
    x=[4, 3.5, 4.5], y=[4.5, 4, 3.5],
    mode="markers+text",
    marker=dict(size=20, color="#ed7a00", opacity=0.7),
    text=["Digital onboarding<br>for offline", "VIP loyalty<br>program", "Fiber optic<br>experience fix"],
    textposition="middle right", textfont=dict(size=15, color="#1e2333"),
    name="Strategic Bets", hoverinfo="text"
))

# Fill-ins (Low Impact, Easy)
fig_matrix.add_trace(go.Scatter(
    x=[1.5, 2], y=[2, 1.5],
    mode="markers+text",
    marker=dict(size=16, color="#29639d", opacity=0.7),
    text=["Referral credit<br>program", "Billing page<br>upsell banner"],
    textposition="middle right", textfont=dict(size=13, color="#1e2333"),
    name="Fill-ins", hoverinfo="text"
))

# Deprioritize (Low Impact, Hard)
fig_matrix.add_trace(go.Scatter(
    x=[4, 4.5], y=[2, 1.5],
    mode="markers+text",
    marker=dict(size=16, color="#e9d845", opacity=0.5),
    text=["Full payment<br>system overhaul", "Re-segment<br>offline customers"],
    textposition="middle right", textfont=dict(size=13, color="#545d6a"),
    name="Deprioritize", hoverinfo="text"
))

# Quadrant labels
fig_matrix.add_annotation(x=0.5, y=4.8, text="QUICK WINS", showarrow=False,
                          font=dict(size=20, color="#39b14b"), xanchor="left")
fig_matrix.add_annotation(x=3.2, y=4.8, text="STRATEGIC BETS", showarrow=False,
                          font=dict(size=20, color="#ed7a00"), xanchor="left")
fig_matrix.add_annotation(x=0.5, y=2.5, text="FILL-INS", showarrow=False,
                          font=dict(size=16, color="#29639d"), xanchor="left")
fig_matrix.add_annotation(x=3.2, y=2.5, text="DEPRIORITIZE", showarrow=False,
                          font=dict(size=16, color="#e9d845"), xanchor="left")

# Quadrant dividers
fig_matrix.add_hline(y=3, line=dict(color="rgba(128,128,128,0.4)", width=1, dash="dot"))
fig_matrix.add_vline(x=2.8, line=dict(color="rgba(128,128,128,0.4)", width=1, dash="dot"))

fig_matrix.update_layout(
    xaxis=dict(title="Implementation Difficulty →", range=[0, 6],
               showgrid=False, zeroline=False, tickvals=[]),
    yaxis=dict(title="Expected Impact →", range=[0.5, 5.3],
               showgrid=False, zeroline=False, tickvals=[]),
    height=550,
    margin=dict(t=20, b=50, l=60, r=40),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#1e2333"),
    legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5,
               font=dict(size=11)),
    showlegend=True
)
st.plotly_chart(fig_matrix, use_container_width=True)

st.success(
    "**Top 3 priorities for immediate action for Cluster 0:** "
    "① Contract migration discount."
    "② Payment auto-pay incentive· "
    "③ Early tenure retention call."
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