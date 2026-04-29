import streamlit as st
import pandas as pd
import backend 
import time
import plotly.express as px 

st.set_page_config(page_title="Batch Customer Prediction", layout="wide")

# Inject custom CSS
st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #161A25;
        border: 1px solid #2D3243;
        padding: 5% 10%;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("## Batch Prediction Dashboard")
st.caption("Upload a CSV file containing multiple customer data and generate churn predictions at scale")
st.divider()

# 1. FILE UPLOADER
st.markdown("### 1. Upload Customer Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

st.info("Make sure CSV has the same columns used in training model.")

if uploaded_file is not None:
    # Read the CSV
    df = pd.read_csv(uploaded_file)
    
    st.success(f"Successfully loaded {len(df)} customers!")
    with st.expander("Preview Uploaded Data"):
        st.dataframe(df.head())

    # 2. RUN BATCH PREDICTION
    st.markdown("### 2. Generate Predictions")
    if st.button("Run Batch Prediction", type="primary"):
        
        # Create empty lists to hold our new predictions
        risk_levels = []
        churn_probs = []
        segments = []
        
        # Create a progress bar
        progress_text = "Analyzing customers. Please wait..."
        my_bar = st.progress(0, text=progress_text)
        
        # Loop through every row in the uploaded CSV
        for index, row in df.iterrows():
            # Convert the row to a dictionary so the backend can read it
            customer_dict = row.to_dict()
            
            # Send to your powerful engine!
            try:
                results = backend.predict_customer(customer_dict)
                risk_levels.append(results['risk_level'])
                churn_probs.append(f"{results['churn_probability'] * 100:.1f}%")
                segments.append(results['cluster_name'])
            except Exception as e:
                # If a row has missing or bad data, mark it as an error so the app doesn't crash
                risk_levels.append("Error")
                churn_probs.append("Error")
                segments.append("Error")
            
            # Update the progress bar
            progress = (index + 1) / len(df)
            my_bar.progress(progress, text=progress_text)
            
        # 3. DISPLAY & DOWNLOAD RESULTS
        # Attach the predictions back to the original dataframe
        df['Predicted Risk Level'] = risk_levels
        df['Churn Probability'] = churn_probs
        df['Customer Segment'] = segments
        
        my_bar.empty() # Clear the progress bar
        st.success("Batch Prediction Complete!")

        # KPI Cards
        st.markdown("### Batch Summary")

        col1, col2, col3, col4 = st.columns(4)

        high_risk_count = len(df[df['Predicted Risk Level'] == 'High'])
        medium_risk_count = len(df[df['Predicted Risk Level'] == 'Medium'])
        low_risk_count = len(df[df['Predicted Risk Level'] == 'Low'])

        col1.metric("Total Customers", len(df))
        col2.metric("High Risk", high_risk_count)
        col3.metric("Medium Risk", medium_risk_count)
        col4.metric("Low Risk", low_risk_count)

        st.divider()

        # Horizontal bar chart
        import plotly.express as px

        risk_counts = df["Predicted Risk Level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]

        fig = px.bar(
             risk_counts,
             x="Count",
             y="Risk Level",
             orientation="h",
             text="Count",
             title="Customer Risk Distribution",
             color="Risk Level",
             color_discrete_map={
             "High": "#ff4b4b",
             "Medium": "#f5c542",
             "Low": "#3fb950"
             }
        )

        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Show the final table
        st.dataframe(df)
        
        # Create a download button for the new CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='churn_predictions_results.csv',
            mime='text/csv',
            type="primary"
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