import streamlit as st
import pandas as pd
import numpy as np
from woe_iv_utils import calculate_iv
import altair as alt

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

    df[target] = pd.to_numeric(
        df[target],
        errors="coerce"
    )

    st.write("Target Distribution")
    st.write(df[target].value_counts(dropna=False))
    
    st.write("Target Dtype")
    st.write(df[target].dtype)
    
    st.write("Unique Target Values")
    st.write(df[target].unique())

    iv_df, woe_df = calculate_iv(
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

    st.subheader("WOE Details")
    st.dataframe(
        woe_df,
        use_container_width=True
    )
    selected_var = st.selectbox(
    "Select Variable for WOE Plot",
    woe_df["Variable"].unique()
)
    
    plot_df = woe_df[
        woe_df["Variable"] == selected_var
    ].copy()
    
    # Find the bin column
    bin_col = [c for c in plot_df.columns
               if c not in [
                   "Variable",
                   "total",
                   "bad",
                   "good",
                   "good_dist",
                   "bad_dist",
                   "WOE",
                   "IV"
               ]][0]
    
    plot_df[bin_col] = (
        plot_df[bin_col]
        .astype(str)
    )
    
    chart = alt.Chart(
        plot_df
    ).mark_bar().encode(
        x=alt.X(
            bin_col,
            sort=None,
            title="Bin"
        ),
        y=alt.Y(
            "WOE",
            title="Weight of Evidence"
        ),
        tooltip=[
            bin_col,
            "WOE",
            "IV"
        ]
    )
    
    st.altair_chart(
        chart,
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
