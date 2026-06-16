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
- WOE IV Analysis
- Model Explainability
- Customer Scoring
- Model Monitoring
- Best Model Hyperparameter Tuning 
- Business Insights
""")

import pandas as pd

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "df" in st.session_state:
        st.sidebar.success(
        f"Dataset Loaded: {st.session_state['df'].shape[0]} rows, "
        f"{st.session_state['df'].shape[1]} columns"
    )

    if "duration" in df.columns:
        df = df.drop("duration", axis=1)

    if "deposit" in df.columns:
        df["deposit"] = df["deposit"].map({
            "no":0,
            "yes":1
        })

    st.session_state["df"] = df

    st.success("Dataset Loaded")
