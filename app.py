import streamlit as st

st.set_page_config(
    page_title="Amex Propensity Modeling Platform",
    layout="wide"
)

st.title(
    "Customer Acquisition Risk & Propensity Modeling Platform"
)

st.markdown("""
### Features

- Data Quality Analysis
- Model Training
- Model Explainability
- Customer Scoring
- Model Monitoring
- Best Model Hyperparameter Tuning  
""")

import streamlit as st
import pandas as pd

st.title("Credit Risk Scorecard Platform")

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    if "duration" in df.columns:
        df = df.drop("duration", axis=1)

    if "deposit" in df.columns:
        df["deposit"] = df["deposit"].map({
            "no":0,
            "yes":1
        })

    st.session_state["df"] = df

    st.success("Dataset Loaded")
