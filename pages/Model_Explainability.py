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
X_sample = X_transformed_df.sample(
    sample_size,
    random_state=42
)
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
tab1, tab2 = st.tabs(
    [
        "Summary Plot",
        "Bar Importance"
    ])
# ----------------------------------------------------------
# SUMMARY PLOT
# ----------------------------------------------------------
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
st.success(
    f"Best Model: {best_model_name}"
)