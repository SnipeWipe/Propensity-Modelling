import streamlit as st
import pandas as pd

st.title("Customer Scoring Dashboard")

# Check model exists
if "trained_models" not in st.session_state:

    st.warning(
        "Train models first"
    )
    st.stop()

# Load best model
trained_models = st.session_state["trained_models"]

best_model_name = st.session_state[
    "best_model_name"
]

best_model = trained_models[
    best_model_name
]

# Upload scoring file
prediction_file = st.file_uploader(
    "Upload Scoring Dataset",
    type="csv"
)

# Score customers
if prediction_file:

    scoring_df = pd.read_csv(
        prediction_file
    )

    expected_columns = st.session_state[
    "X"
    ].columns.tolist()

    missing_cols = list(
        set(expected_columns)
        -
        set(scoring_df.columns)
    )

    if missing_cols:

        st.error(
            f"Missing columns: {missing_cols}"
        )

        st.stop()

    scores = best_model.predict_proba(
        scoring_df
    )[:,1]

    scoring_df["Propensity_Score"] = scores

    # KPI Dashboard
    col1,col2,col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Customers",
            len(scoring_df)
        )

    with col2:
        st.metric(
            "Average Propensity",
            f"{scores.mean():.2%}"
        )

    with col3:
        st.metric(
            "High Propensity Customers",
            (scores > 0.7).sum()
        )

    # Distribution Chart
    st.subheader(
        "Propensity Distribution"
    )

    distribution = (
    pd.cut(
        scoring_df["Propensity_Score"],
        bins=[0,0.2,0.4,0.6,0.8,1],
        labels=[
            "0-20%",
            "20-40%",
            "40-60%",
            "60-80%",
            "80-100%"]).value_counts().sort_index())

    distribution_df = pd.DataFrame({
    "Score Band": distribution.index,
    "Customers": distribution.values})

    st.bar_chart(
    distribution_df.set_index("Score Band"))

    # Show Scored Customers
    st.dataframe(
        scoring_df,
        height=500,
        use_container_width=True
    )

    # Download Predictions
    csv = scoring_df.to_csv(
        index=False
    )

    st.download_button(
        "Download Predictions",
        csv,
        "predictions.csv",
        "text/csv"
    )