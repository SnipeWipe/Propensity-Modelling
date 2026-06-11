import streamlit as st
import pandas as pd

from utils.woe_iv_utils import calculate_iv

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

    st.download_button(
        "Download IV Report",
        iv_df.to_csv(index=False),
        file_name="iv_report.csv",
        mime="text/csv"
    )
