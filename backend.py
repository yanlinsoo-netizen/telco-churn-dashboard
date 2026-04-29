import pandas as pd
import joblib
import json
import shap
import streamlit as st

# 1. Create a cached function to load everything into memory once
@st.cache_resource
def load_all_models():
    """Loads and caches all machine learning models and configurations."""
    
    # Classification Files
    rf = joblib.load('rf_model.pkl')
    enc = joblib.load('encoder.pkl')
    scl = joblib.load('scaler.pkl')
    shap_exp = joblib.load('shap_explainer.pkl')

    with open('feature_names.json', 'r') as f:
        rf_feat = json.load(f)

    # Clustering Files
    kmeans = joblib.load('kmeans_model.pkl')
    kmeans_prep = joblib.load('kmeans_preprocessor.pkl')

    with open('kmeans_feature_names.json', 'r') as f:
        kmeans_feat = json.load(f)
        
    return rf, enc, scl, shap_exp, rf_feat, kmeans, kmeans_prep, kmeans_feat

# 2. Unpack the cached models into the global variables your script already uses
(rf_model, encoder, scaler, shap_explainer, rf_feature_names, 
 kmeans_model, kmeans_preprocessor, kmeans_feature_names) = load_all_models()

# Master prediction function
def predict_customer(customer_dict: dict) -> dict:
    """Processes a single customer dictionary through the entire ML pipeline."""
    
    # Convert input dict to a pandas DataFrame
    df_input = pd.DataFrame([customer_dict])
    
    # Part A: Clustering (K-Means)
    # 1. Apply get_dummies
    df_cluster = pd.get_dummies(df_input, drop_first=True)
    
    # 2. Ensure columns match the clustering training data perfectly
    for col in kmeans_feature_names:
        if col not in df_cluster.columns:
            df_cluster[col] = 0
    df_cluster = df_cluster[kmeans_feature_names] 
    
    # 3. Preprocess and predict
    cluster_scaled = kmeans_preprocessor.transform(df_cluster)
    cluster_label = kmeans_model.predict(cluster_scaled)[0]
    
    # Map cluster number to defined profiles
    cluster_map = {
        0: "High Churn (Critical)",
        1: "Loyal Premium (Best)",
        2: "Basic Offline (Stable)"
    }
    cluster_name = cluster_map.get(cluster_label, f"Cluster {cluster_label}")

    # Part B: Classification(Random Forest)
    # 1. Identify columns
    categorical_cols = [
        'Gender', 'Senior Citizen', 'Partner', 'Dependents', 
        'Phone Service', 'Multiple Lines', 'Internet Service', 
        'Online Security', 'Online Backup', 'Device Protection', 
        'Tech Support', 'Streaming TV', 'Streaming Movies', 
        'Contract', 'Paperless Billing', 'Payment Method'
    ]
    numerical_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'CLTV']
    
    # Ensure numerical columns exist, default to 0 if missing
    for col in numerical_cols:
        if col not in df_input.columns:
            df_input[col] = 0.0

    # 2. Apply One-Hot Encoding
    encoded_array = encoder.transform(df_input[categorical_cols])
    new_cat_names = encoder.get_feature_names_out(categorical_cols)
    df_encoded = pd.DataFrame(encoded_array, columns=new_cat_names, index=df_input.index)
    
    # 3. Swap text columns for encoded columns
    df_class = df_input.drop(columns=categorical_cols).join(df_encoded)
    
    # 4. Scale numerical columns
    df_class[numerical_cols] = scaler.transform(df_class[numerical_cols])
    
    # 5. Drop the 6 redundant columns & ensure exact order
    for col in rf_feature_names:
        if col not in df_class.columns:
            df_class[col] = 0
    df_final = df_class[rf_feature_names] 
    
    # 6. Predict churn probability
    churn_prob = rf_model.predict_proba(df_final)[0][1] 
    churn_label = "Yes" if churn_prob > 0.5 else "No"
    
    # 7. Assign risk level
    if churn_prob >= 0.7:
        risk_level = "High"
    elif churn_prob >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"
        
    # 8. Generate SHAP values for explainability
    shap_values = shap_explainer(df_final)
    
    # Return package
    return {
        'churn_probability': float(churn_prob),
        'churn_label': churn_label,
        'risk_level': risk_level,
        'cluster': int(cluster_label),
        'cluster_name': cluster_name,
        'shap_values': shap_values 
    }