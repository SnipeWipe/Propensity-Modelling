import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib

st.title("Model Explainability")

if "trained_models" not in st.session_state:
    st.warning("Train models first")
    st.stop()

best_model_name = st.session_state["best_model_name"]
best_model = st.session_state["trained_models"][best_model_name]
X_test = st.session_state["X_test"]

# Get the model and preprocessor
model = best_model.named_steps["model"]
preprocessor = best_model.named_steps["preprocessor"]

# === Get exact feature names (robust version) ===
try:
    feature_names = preprocessor.get_feature_names_out()
except Exception:
    # Fallback: try to reconstruct names if get_feature_names_out fails
    try:
        feature_names = X_test.columns.tolist()
    except:
        feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]

# Feature importance
if hasattr(model, "feature_importances_"):
    importance = model.feature_importances_
elif hasattr(model, "coef_"):
    importance = np.abs(
        model.coef_.ravel()
    )
else:
    st.warning(
        "Feature importance unavailable"
    )
    st.stop()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values("Importance", ascending=False).head(20)

st.subheader("Top 20 Most Important Features")
st.bar_chart(importance_df.set_index("Feature"))

import numpy as np
X_test_display = X_test.reset_index(
    drop=True
)

X_sample_display = X_sample.reset_index(
    drop=True
)
# ==========================================================
# SHAP ANALYSIS
# ==========================================================
st.subheader("SHAP Explainability")
# Transform dataset
X_transformed = preprocessor.transform(X_test)
# Convert sparse matrix to dense
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()
# Create dataframe with proper feature names
X_transformed_df = pd.DataFrame(
    X_transformed,
    columns=feature_names
)
# Sample data for performance
sample_size = min(500, len(X_transformed_df))
X_sample = (
    X_transformed_df.sample(
        sample_size,
        random_state=42).reset_index(drop=True)
)
X_sample = X_sample.reset_index(drop=True)
# ==========================================================
# CREATE SHAP EXPLAINER
# ==========================================================
try:
    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(
            model
        )
        shap_values = explainer.shap_values(
            X_sample
        )
        if isinstance(shap_values, list):
            customer_shap = shap_values[1]
        else:
            customer_shap = shap_values
    elif hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(
            model,
            X_sample
        )
        shap_values = explainer.shap_values(
            X_sample
        )
    else:
        explainer = shap.Explainer(
            model.predict,
            X_sample
        )
        shap_values = explainer(
            X_sample
        )
except Exception as e:
    st.error(
        f"SHAP Error: {e}"
    )
    st.stop()
# ==========================================================
# HANDLE BINARY CLASSIFICATION OUTPUT
# ==========================================================
if isinstance(shap_values, list):
    shap_values = shap_values[1]
elif hasattr(shap_values, "values"):
    shap_values = shap_values.values
# ==========================================================
# SHAP VISUALIZATION TABS
# ==========================================================
tab1, tab2, tab3 = st.tabs(
    [
        "Summary Plot",
        "Bar Importance",
        "Customer Level Explanation"
    ])
# ----------------------------------------------------------
# SUMMARY PLOT
# ----------------------------------------------------------
customer_summary = pd.DataFrame({
    "Customer ID": [
        f"CUST_{i:05d}"
        for i in range(len(X_sample))
    ],
    "Predicted Score": best_model.predict_proba(
        X_test
    )[:,1]
})
    st.dataframe(
        customer_summary.head(100)
    )

with tab1:
    st.write(
        "Global SHAP Impact"
    )
    fig1 = plt.figure(
        figsize=(12,8)
    )
    shap.summary_plot(
        shap_values,
        X_sample,
        show=False
    )
    st.pyplot(fig1)
    plt.close(fig1)
# ----------------------------------------------------------
# BAR PLOT
# ----------------------------------------------------------
with tab2:
    st.write(
        "Mean Absolute SHAP Values"
    )
    fig2 = plt.figure(
        figsize=(12,8)
    )
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False
    )
    st.pyplot(fig2)
    plt.close(fig2)

with tab3:

    st.subheader(
        "Customer-Level Explanation")
    
    selected_customer = st.selectbox(
    "Choose Customer",
    customer_summary["Customer ID"]
)

    # Prediction Score
    customer_data = X_test.iloc[
        [customer_idx]
    ]

    score = best_model.predict_proba(
        customer_data
    )[0,1]

    st.metric(
        "Predicted Propensity",
        f"{score:.2%}"
    )

    # Waterfall Plot
    st.subheader(
        "Why Did The Model Predict This Score?"
    )

    fig = plt.figure(
        figsize=(10,6)
    )

    shap.plots.waterfall(
        shap.Explanation(
            values=customer_shap[
                customer_idx
            ],
            base_values=explainer.expected_value
            if not isinstance(
                explainer.expected_value,
                list
            )
            else explainer.expected_value[1],
            data=X_sample.iloc[
                customer_idx
            ],
            feature_names=X_sample.columns
        ),
        show=False
    )

    st.pyplot(fig)

    plt.close(fig)

sample_idx = X_sample.index[customer_idx]

customer_data = X_test.iloc[
    [sample_idx]
]

score = best_model.predict_proba(
    customer_data
)[0,1]

st.metric(
    "Predicted Propensity",
    f"{score:.2%}"
)

contrib_df = pd.DataFrame({
    "Feature": X_sample.columns,
    "SHAP Value": customer_shap[
        customer_idx
    ]
})

positive_df = (
    contrib_df
    .sort_values(
        "SHAP Value",
        ascending=False
    )
    .head(10)
)

negative_df = (
    contrib_df
    .sort_values(
        "SHAP Value"
    )
    .head(10)
)

col1,col2 = st.columns(2)

with col1:
    st.subheader(
        "Top Positive Drivers"
    )
    st.dataframe(
        positive_df
    )

with col2:
    st.subheader(
        "Top Negative Drivers"
    )
    st.dataframe(
        negative_df
    )
st.success(
    f"Best Model: {best_model_name}"
)
