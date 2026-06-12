import streamlit as st
import pandas as pd

from woe_iv_utils import calculate_iv

st.title("WOE & IV Analysis")

if "df" not in st.session_state:
    st.warning("Upload dataset first")
    st.stop()

df = st.session_state["df"]

target = st.selectbox(
    "Select Target",
    df.columns
)

if st.button("Run IV Analysis"):

    df = df.copy()
    df[target] = df[target].map({
        "no": 0,
        "yes": 1
    })

    df[target] = pd.to_numeric(
        df[target],
        errors="coerce"
    )

    st.write(feature)
    st.write(grouped.head())

    st.write("Target Distribution")
    st.write(df[target].value_counts())
    
    st.write("Target Dtype")
    st.write(df[target].dtype)

    iv_df = calculate_iv(
        df,
        target
    )

    iv_df = iv_df.sort_values(
        by="info_value",
        ascending=False
    )

    st.subheader("Information Value")

    st.dataframe(
        iv_df,
        use_container_width=True
    )
    
    st.markdown("""
    ### IV Interpretation
    
    | IV Value | Predictive Power |
    |-----------|------------------|
    | < 0.02 | Not useful |
    | 0.02 - 0.1 | Weak |
    | 0.1 - 0.3 | Medium |
    | 0.3 - 0.5 | Strong |
    | > 0.5 | Suspiciously Powerful |
    """)

    st.download_button(
        "Download IV Report",
        iv_df.to_csv(index=False),
        file_name="iv_report.csv",
        mime="text/csv"
    )
