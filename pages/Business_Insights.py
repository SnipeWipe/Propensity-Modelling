import streamlit as st
import pandas as pd

st.title("Business Insights")

if "results_df" not in st.session_state:
    st.warning(
        "Train model first"
    )
    st.stop()

results = st.session_state[
    "results_df"
]

best_model = results.sort_values(
    "AUC",
    ascending=False
).iloc[0]

col1,col2,col3 = st.columns(3)

col1.metric(
    "Best AUC",
    round(best_model["AUC"],4)
)

col2.metric(
    "Best KS",
    round(best_model["KS"],4)
)

col3.metric(
    "Best Gini",
    round(best_model["Gini"],4)
)
