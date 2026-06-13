import streamlit as st
import pandas as pd

st.title("Customer Scoring Dashboard")
st.write("Training Features")
st.write(st.session_state["X"].columns.tolist())
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
    try:
        scoring_df = pd.read_csv(prediction_file)
        st.success("Scoring file uploaded successfully")
        # ==========================================
        # Training Schema
        # ==========================================
        expected_columns = st.session_state["X"].columns.tolist()
        num_cols = st.session_state["num_cols"]
        cat_cols = st.session_state["cat_cols"]
        # ==========================================
        # Missing Columns
        # ==========================================
        missing_cols = [
            col for col in expected_columns
            if col not in scoring_df.columns
        ]
        if missing_cols:
            st.warning(
                f"Missing columns detected: {missing_cols}"
            )
            # Create missing columns
            for col in missing_cols:
                if col in num_cols:
                    scoring_df[col] = 0
                else:
                    scoring_df[col] = "Unknown"
        # ==========================================
        # Remove Extra Columns
        # ==========================================
        extra_cols = [
            col for col in scoring_df.columns
            if col not in expected_columns
        ]
        if extra_cols:
            st.info(
                f"Ignoring extra columns: {extra_cols}"
            )
            scoring_df = scoring_df.drop(
                columns=extra_cols
            )
        # ==========================================
        # Reorder Columns
        # ==========================================
        scoring_df = scoring_df[
            expected_columns
        ]
        # ==========================================
        # Convert Numeric Columns
        # ==========================================
        for col in num_cols:
            scoring_df[col] = pd.to_numeric(
                scoring_df[col],
                errors="coerce"
            )
        # ==========================================
        # Handle Missing Values
        # ==========================================
        for col in num_cols:
            scoring_df[col] = scoring_df[col].fillna(
                scoring_df[col].median()
            )
        for col in cat_cols:
            scoring_df[col] = scoring_df[col].fillna(
                "Unknown"
            )
        # ==========================================
        # Predict
        # ==========================================
        scores = best_model.predict_proba(
            scoring_df
        )[:, 1]
        scoring_df["Propensity_Score"] = scores
        # =========================================
        # Dashboard Metrics
        # ==========================================
        col1, col2, col3 = st.columns(3)
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
        # ==========================================
        # Distribution
        # ==========================================
        st.subheader(
            "Propensity Distribution"
        )
        distribution = pd.cut(
            scoring_df["Propensity_Score"],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
            labels=[
                "0-20%",
                "20-40%",
                "40-60%",
                "60-80%",
                "80-100%"
            ]
        ).value_counts().sort_index()

        distribution_df = pd.DataFrame({
            "Score Band": distribution.index,
            "Customers": distribution.values
        })
        st.bar_chart(
            distribution_df.set_index(
                "Score Band"
            )
        )

        st.subheader("Schema Validation")
        validation_df = pd.DataFrame({
            "Column": expected_columns,
            "Present": [
                col in scoring_df.columns
                for col in expected_columns
            ]
        })
        st.dataframe(validation_df)

        scoring_df["Segment"] = pd.cut(
        scoring_df["Propensity_Score"],
        bins=[0,0.3,0.6,1],
        labels=[
            "Low",
            "Medium",
            "High"
        ]
    )
        
        st.subheader("Top 100 Customers")
        st.dataframe(
            scoring_df.sort_values(
                "Propensity_Score",
                ascending=False
            ).head(100)
        )

        st.subheader("Customers Scoring Prediction")
        st.dataframe(
            scoring_df,
            use_container_width=True,
            height=500
        )
        csv = scoring_df.to_csv(
            index=False
        )
        st.download_button(
            "Download Predictions",
            csv,
            "predictions.csv",
            "text/csv"
        )
    except Exception as e:
        st.error(
            f"Scoring Failed: {str(e)}"
        )
