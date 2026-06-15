import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

st.title("Customer Propensity Modeling Dashboard")

if "df" not in st.session_state:
    st.warning("Please upload a dataset first in the Data Quality Page.")
    st.stop()

df = st.session_state["df"] 
st.dataframe(df.head(), use_container_width=True)

target = st.selectbox("Select Target Variable", df.columns)
st.session_state["target"] = target

if st.button("Train Models"):
    y = df[target]
    
    if y.nunique() != 2:
        st.error("Target must be binary.")
        st.stop()

    X = df.drop(columns=[target])
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_cols = X.select_dtypes(exclude='object').columns.tolist()

    preprocessor = ColumnTransformer([
    ('num',Pipeline([
            ('imputer',
             SimpleImputer(strategy='median')),
            ('scaler',
             StandardScaler())
        ]),num_cols),
    ('cat',Pipeline([
            ('imputer',
             SimpleImputer(strategy='most_frequent')),
            ('encoder',
             OneHotEncoder(handle_unknown='ignore'))
        ]),cat_cols)
])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
    }

    results = []
    trained_models = {}

    fig_roc, ax_roc = plt.subplots(figsize=(10, 7))
    fig_pr, ax_pr = plt.subplots(figsize=(10, 7))

    for name, model in models.items():
        pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        trained_models[name] = pipe

        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]

        # === Metrics ===
        auc = roc_auc_score(y_test, prob)
        gini = 2 * auc - 1
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test,pred,average="weighted",zero_division=0)
        rec = recall_score(y_test,pred,average="weighted",zero_division=0)
        f1 = f1_score(y_test,pred,average="weighted",zero_division=0)

        # === KS Statistic (Calculated for EVERY model) ===
        ks_df = pd.DataFrame({"target": y_test, "score": prob})
        ks_df = ks_df.sort_values("score", ascending=False).reset_index(drop=True)
        ks_df["bucket"] = pd.qcut(ks_df["score"],10,labels=False,duplicates="drop")
        grouped = ks_df.groupby("bucket")
        cum_good = grouped["target"].apply(lambda x: (x == 0).sum()).cumsum() / (y_test == 0).sum()
        cum_bad = grouped["target"].apply(lambda x: (x == 1).sum()).cumsum() / (y_test == 1).sum()
        ks_value = max(abs(cum_bad - cum_good))

        results.append([name, acc, prec, rec, f1, auc,gini, ks_value])

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

        # PR Curve
        precision, recall, _ = precision_recall_curve(y_test, prob)
        ax_pr.plot(recall, precision, label=f"{name} (P={prec:.3f}, R={rec:.3f})")

    # Finalize comparison plots
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax_roc.legend()
    ax_pr.legend()

    # === Results DataFrame with KS ===
    results_df = pd.DataFrame(
        results, 
        columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC','GINI', 'KS']
    )

    # ================== BEST MODEL SELECTION ==================
    # Best model is chosen based on highest AUC (most stable metric)
    best_model_name = results_df.sort_values("AUC", ascending=False).iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    os.makedirs(
    "models",
    exist_ok=True)

    joblib.dump(
        best_model,
        "models/best_model.pkl"
    )
    best_ks = results_df[results_df["Model"] == best_model_name]["KS"].values[0]

    st.success(f"**Best Model:** {best_model_name} | AUC = {results_df['AUC'].max():.4f} | KS = {best_ks:.4f}")

    # ================== TABS ==================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Model Comparison", "ROC & PR Curves", "Confusion Matrix", "Lift & KS", "Feature Importance"
    ])

    # ---------------- TAB 1: Model Comparison ----------------
    with tab1:
        st.subheader("Model Performance Comparison")
        st.dataframe(results_df.style.highlight_max(subset=['AUC', 'KS'], color='lightgreen'), 
                     use_container_width=True)

    # ---------------- TAB 2: ROC & PR Curves ----------------
    with tab2:
        st.subheader("ROC Curve Comparison")
        st.pyplot(fig_roc)
        st.subheader("Precision-Recall Curve Comparison")
        st.pyplot(fig_pr)

    # ---------------- TAB 3: Confusion Matrix ----------------
    with tab3:
        st.subheader(f"Confusion Matrix - {best_model_name}")
        best_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, best_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        st.pyplot(fig_cm)

        st.subheader("Classification Report")
        report = classification_report(y_test, best_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    # ---------------- TAB 4: Lift & KS (Best Model) ----------------
    with tab4:
        st.subheader(f"Lift Chart - {best_model_name}")
        best_prob = best_model.predict_proba(X_test)[:, 1]
        risk_df = pd.DataFrame({
            "Probability": best_prob
        })
        
        risk_df["Risk Segment"] = pd.cut(
            risk_df["Probability"],
            bins=[0,0.3,0.7,1],
            labels=[
                "Low Risk",
                "Medium Risk",
                "High Risk"
            ]
        )
        
        st.subheader("Risk Segment Distribution")
        
        st.dataframe(
            risk_df["Risk Segment"]
            .value_counts()
            .reset_index(),
            use_container_width=True
        )
        
        decile_df = pd.DataFrame({
            "Actual": y_test,
            "Probability": best_prob
        })
        
        decile_df = decile_df.sort_values(
            "Probability",
            ascending=False
        ).reset_index(drop=True)
        
        decile_df["Decile"] = pd.qcut(
            decile_df.index,
            10,
            labels=False
        )
        
        decile_summary = decile_df.groupby(
            "Decile"
        ).agg(
            Customers=("Actual","count"),
            Responders=("Actual","sum"),
            Response_Rate=("Actual","mean")
        ).reset_index()
        
        st.subheader("Decile Analysis")
        
        st.dataframe(
            decile_summary,
            use_container_width=True
        )
        lift_df = pd.DataFrame({"Actual": y_test, "Score": best_prob})
        lift_df = lift_df.sort_values("Score", ascending=False).reset_index(drop=True)
        lift_df["Decile"] = pd.qcut(range(len(lift_df)), 10, labels=False, duplicates='drop')
        lift = lift_df.groupby("Decile")["Actual"].mean()

        fig_lift, ax_lift = plt.subplots(figsize=(10, 6))
        lift.plot(kind="bar", ax=ax_lift, color="skyblue")
        ax_lift.axhline(y=y_test.mean(), color='red', linestyle='--', label='Overall Response Rate')
        ax_lift.set_ylabel("Response Rate")
        ax_lift.legend()
        st.pyplot(fig_lift)

        st.metric("KS Statistic (Best Model)", round(best_ks, 4))

    # ---------------- TAB 5: Feature Importance + CV ----------------
    with tab5:
        st.subheader("Feature Importance")
        model_step = best_model.named_steps["model"]
        if hasattr(model_step, "feature_importances_"):
            feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
            importances = model_step.feature_importances_
            feat_imp = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values("Importance", ascending=False)
            st.dataframe(feat_imp.head(20), use_container_width=True)
        else:
            st.info(f"{best_model_name} does not support feature importance.")

        st.subheader("Cross Validation Performance")
        cv_auc = cross_val_score(best_model, X, y, cv=5, scoring="roc_auc").mean()
        st.metric("Mean CV AUC (5-Fold)", f"{cv_auc:.4f}")

    # Save to session state
    st.session_state["best_model"] = best_model
    st.session_state["trained_models"] = trained_models
    st.session_state["preprocessor"] = preprocessor
    st.session_state["results_df"] = results_df
    st.session_state["best_model_name"] = best_model_name
    st.session_state["X"] = X
    st.session_state["num_cols"] = num_cols
    st.session_state["cat_cols"] = cat_cols
    st.session_state["df"] = df
    st.session_state["X_train"] = X_train
    st.session_state["y_train"] = y_train
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test

    st.success("Models Trained Successfully!")
    st.session_state["training_data"] = X_train.copy()
