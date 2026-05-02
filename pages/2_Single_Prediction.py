import streamlit as st
import matplotlib.pyplot as plt
import shap
import backend

st.set_page_config(page_title="Single Customer Prediction", layout="wide")

# Inject CSS
st.markdown("""
<style>
    /* 1. Style the Metric Cards: Premium Dark Accents */
    div[data-testid="stMetric"] {
        background-color: #1A2235 !important; /* Forces the deep dark slate background */
        border: 1px solid #2D3243 !important;
        border-left: 5px solid #4da6ff !important; /* Sharp blue accent line */
        padding: 15px 20px !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15) !important; 
    }
    
    /* Sharp, crisp label text */
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] label p {
        color: #8b949e !important; 
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important; 
        letter-spacing: 1px !important;
    }

    /* Giant, bright white value text */
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 2.4rem !important; 
        font-weight: 800 !important; 
        color: #ffffff !important; 
    }

    /* 2. Keep the Main Input Form Box DARK BLUE */
    div[data-testid="stForm"] {
        background-color: #1A2235 !important; 
        border: 1px solid #2D3243;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }

    /* 3. Force the words INSIDE the dark form to be WHITE */
    div[data-testid="stForm"] label, 
    div[data-testid="stForm"] label p {
        color: #ffffff !important; 
    }
    div[data-testid="stForm"] h3, 
    div[data-testid="stForm"] h2 {
        color: #ffffff !important; 
    }
    div[data-testid="stForm"] hr {
        border-color: #4da6ff !important; 
    }

    /* 4. Make the Action Plan Cards "Float" with Hover Effects */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #4da6ff;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
        transform: translateY(-3px);
    }
</style>
""", unsafe_allow_html=True)

st.title("Single Customer Prediction")
st.write("Enter the customer's details below to predict their churn risk and segment profile.")

# 1. Build the input form
with st.form("customer_input_form"):
    st.subheader("Customer Demographics & Account")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
       tenure = st.number_input("Tenure Months", min_value=0, max_value=200, value=12)
       gender = st.selectbox("Gender", ["Female", "Male"])
    
    with col2:
       monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=50.0)
       senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    
    with col3:
       total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)
       partner = st.selectbox("Partner", ["No", "Yes"])
    
    with col4:
       cltv = st.number_input("CLTV (Customer Lifetime Value)", min_value=0, value=4000)
       dependents = st.selectbox("Dependents", ["No", "Yes"])

    st.markdown("---")
    st.subheader("Subscribed Services")
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
    with col_s2:
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        
    with col_s3:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.markdown("---")
    st.subheader("Billing & Contract")
    col_b1, col_b2, col_b3 = st.columns(3)
    
    with col_b1:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    
    with col_b2:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    
    with col_b3:
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", 
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    # The submit button
    submitted = st.form_submit_button("Predict Churn Risk", type="primary", use_container_width=True)


# 2. Handle the prediction
if submitted:
    # Package the inputs exactly as the backend expects
    customer_data = {
        "Tenure Months": tenure,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges,
        "CLTV": cltv,
        "Gender": gender,
        "Senior Citizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Contract": contract,
        "Paperless Billing": paperless,
        "Payment Method": payment_method,
        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet_service,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies
    }
    
    with st.spinner("Analyzing customer data..."):
        # Send to backend
        results = backend.predict_customer(customer_data)
    
    # Phase 1: Clustering
    st.markdown("---")
    st.subheader("1. Customer Profile (Segmentation)")
    st.write("Based on their billing and service attributes, this customer aligns with the following historical group:")
    st.info(f"**{results['cluster_name']}**")
    
    # Phase 2: Predictive ML (Random Forest)
    st.markdown("---")
    st.subheader("2. Targeted Churn Risk")
    st.write("Our Random Forest model calculated this individual's specific probability of churning:")
    
    # Phase 3: Display top-level metrics
    r1, r2 = st.columns(2)
    r1.metric("Churn Risk Level", results['risk_level'])
    r2.metric("Probability of Churn", f"{results['churn_probability'] * 100:.1f}%")
    
    # Phase 4: SHAP
    st.write("**Why did the model make this decision?**")
    plt.rcParams.update({'text.color': 'black', 'axes.labelcolor': 'black',
                         'xtick.color': 'black', 'ytick.color': 'black'})
    fig, ax = plt.subplots(figsize=(3, 2))
    shap.plots.waterfall(results['shap_values'][0, :, 1], show=False)
    
    # Create 3 columns: Empty space (15%), Chart (70%), Empty space (15%)
    spacer1, chart_col, spacer2 = st.columns([0.15, 0.70, 0.15])
    
    # Put the chart inside the middle column
    with chart_col:
        st.pyplot(fig, bbox_inches='tight', transparent=True)

    # Phase 5: Action plan (Driven by ML Risk)
    st.markdown("---")
    st.subheader("Recommended Action Plan")

    risk = results['risk_level']

    if risk == "High":
        # Custom HTML banner for Urgent Risk
        st.markdown("""
        <div style="background-color: rgba(255, 75, 75, 0.1); padding: 15px; border-radius: 8px; border-left: 6px solid #ff4b4b; margin-bottom: 20px;">
            <h4 style="color: #ff4b4b; margin: 0;">⚠️ URGENT ACTION NEEDED</h4>
            <p style="margin: 0; font-size: 14px; color: #0a0d0f;">This customer is at high risk of churning. Execute the following immediately:</p>
        </div>
        """, unsafe_allow_html=True)

        # Create a 2-column grid of cards
        c1, c2 = st.columns(2)
        
        with c1:
            with st.container(border=True):
                st.markdown("#### Contract Migration")
                st.write("Offer a **15-20% discount** for switching from Month-to-Month to a 12-month plan.")
                st.caption("Quick Win • Removes monthly flight risk")
                
            with st.container(border=True):
                st.markdown("#### Payment Auto-Pay")
                st.write("Incentivize switching to Auto-Pay with a **one-time $20 credit**.")
                st.caption("Quick Win • Reduces payment friction")
                
        with c2:
            with st.container(border=True):
                st.markdown("#### Early Intervention")
                st.write("Schedule a **proactive retention call** within the next 7 days.")
                st.caption("Strategic Bet • Addresses immediate friction")

    elif risk == "Medium":
        st.markdown("""
        <div style="background-color: rgba(237, 122, 0, 0.1); padding: 15px; border-radius: 8px; border-left: 6px solid #ed7a00; margin-bottom: 20px;">
            <h4 style="color: #ed7a00; margin: 0;">⚠️ PREVENTATIVE ACTION</h4>
            <p style="margin: 0; font-size: 14px; color: #0a0d0f;">This customer is unstable. Deepen their reliance on our ecosystem:</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown("#### Service Bundle")
                st.write("Offer a **free 3-month trial** of the Online Security + Tech Support bundle.")
                st.caption("Fill-In • Increases platform reliance")
                
            with st.container(border=True):
                st.markdown("#### Payment Modernization")
                st.write("Guide migration from mailed checks to online auto-pay with a **small monthly discount**.")
                st.caption("Quick Win • Modernizes account")
                
        with c2:
            with st.container(border=True):
                st.markdown("#### Digital Onboarding")
                st.write("If they lack internet, offer an entry-level bundle at an **introductory rate**.")
                st.caption("Strategic Bet • Massive upsell potential")

    elif risk == "Low":
        st.markdown("""
        <div style="background-color: rgba(67, 219, 90, 0.1); padding: 15px; border-radius: 8px; border-left: 6px solid #43db5a; margin-bottom: 20px;">
            <h4 style="color: #43db5a; margin: 0;">✅ MAINTAIN & GROW</h4>
            <p style="margin: 0; font-size: 14px; color: #0a0d0f;">This customer is highly stable. Focus on loyalty and premium upgrades:</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown("#### VIP Loyalty")
                st.write("Ensure enrollment in tiered loyalty programs (anniversary rewards, early access).")
                st.caption("Strategic Bet • Validates loyalty")
                
            with st.container(border=True):
                st.markdown("#### Referral Incentive")
                st.write("Offer a **$25 statement credit** for each successful referral.")
                st.caption("Fill-In • Low CAC acquisition")
                
        with c2:
            with st.container(border=True):
                st.markdown("#### Cross-sell Premium")
                st.write("Pitch premium speed upgrades or bundled family streaming packages.")
                st.caption("Quick Win • Increases Lifetime Value")

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
