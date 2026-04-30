# Telco Customer Churn Analytics Dashboard

**WQD7012 Applied Machine Learning Project**

An enterprise-grade, end-to-end machine learning application built with Python and Streamlit. This dashboard provides actionable, data-driven retention strategies using predictive modeling, customer segmentation, and Explainable AI (XAI).

## Live Demo
[**Access the Live Web App Here**](https://)

---

## Project Overview
Customer churn is a critical and expensive problem for telecommunications companies. This project utilizes the foundational IBM Telco Dataset (7043 historical records) to go beyond simply predicting which customers will leave. It explains why they are leaving and recommends specific, cluster-based business interventions.

### Key Features
* **Executive Dashboard:** A high-level command center tracking global churn rates, total revenue lost, and demographic distributions.
* **Real-time Single Prediction:** An interactive form to input customer data, outputting a churn probability and a SHAP waterfall chart that explains the driving factors.
* **Batch Processing:** Upload a CSV of customer records to generate bulk predictions and downloadable risk reports.
* **Customer Segmentation:** Deep-dive profiling using clustering, complete with interactive radar charts.
* **Actionable Recommendations:** SHAP-driven retention strategies mapped onto a Priority Matrix based on expected impact and implementation effort.

---

## Modeling & Methodology
* **Classification Engine:** Random Forest Classifier (Optimized decision threshold: 0.40 to balance recall for high-risk customers).
* **Segmentation Engine:** K-Means Clustering (k=3). Evaluated against K-Medoids and validated via Silhouette Score (0.321), Davies-Bouldin Index, and Calinski-Harabasz Index.
* **Explainable AI (XAI):** SHAP (SHapley Additive exPlanations) values are utilized locally to ensure model transparency.

---

## Running the App Locally
If you wish to run this dashboard on your own machine, follow these steps:

1. Clone the repository:
   git clone https://github.com/yanlinsoo-netizen/telco-churn-dashboard.git
   cd telco-churn-dashboard

2. Install dependencies:
   pip install -r requirements.txt

3. Launch the application:
   streamlit run 1_Dashboard.py

---

## Team Members
Developed collaboratively by Group 6:
* Soo Yan Lin
* Bian ChenFang
* Yah Hong Xuan
* Moo Nee Choong
* Koh Shin Yi
