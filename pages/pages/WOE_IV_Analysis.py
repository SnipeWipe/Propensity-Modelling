import streamlit as st
import pandas as pd
import scorecardpy as sc

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

    iv_df = sc.iv(
        df,
        y=target
    )

    st.subheader(
        "Information Value"
    )

    st.dataframe(
        iv_df.sort_values(
            "info_value",
            ascending=False
        ),
        use_container_width=True
    )
