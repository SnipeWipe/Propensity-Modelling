import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

st.title("Hyperparameter Tuning")
if "best_model" not in st.session_state:
    st.warning(
        "Please train models first."
    )
    st.stop()

#------------------------
#Creating DB
#----------------------
conn = sqlite3.connect("model_registry.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS registry (
    Timestamp TEXT,
    Model TEXT,
    CV_AUC REAL,
    Best_Params TEXT
)
""")

conn.commit()

# --------------------------------------------------
# Load objects
# --------------------------------------------------
best_model = st.session_state["best_model"]
best_model_name = st.session_state[
    "best_model_name"
]
preprocessor = st.session_state[
    "preprocessor"
]
X_train = st.session_state[
    "X_train"
]
y_train = st.session_state[
    "y_train"
]
st.success(
    f"Current Best Model: {best_model_name}"
)
# --------------------------------------------------
# Optional Override
# --------------------------------------------------
use_other_model = st.checkbox(
    "Tune a different model"
)
if use_other_model:
    trained_models = st.session_state[
        "trained_models"
    ]
    selected_model = st.selectbox(
        "Select Model",
        list(trained_models.keys())
    )
    base_model = trained_models[
        selected_model
    ].named_steps["model"]
else:
    selected_model = best_model_name
    base_model = best_model.named_steps[
        "model"
    ]
# --------------------------------------------------
# Parameter Grid
# --------------------------------------------------
param_grids = {
    "Logistic Regression": {
        "model__C": [
            0.01,
            0.1,
            1,
            10
        ]
    },
    "Random Forest": {
        "model__n_estimators": [
            100,
            200,
            500
        ],
        "model__max_depth": [
            5,
            10,
            20
        ]
    },
    "Gradient Boosting": {
        "model__n_estimators": [
            100,
            200
        ],
        "model__learning_rate": [
            0.01,
            0.1
        ]
    },
    "XGBoost": {
        "model__n_estimators": [
            100],
        "model__max_depth": [
            3,
            5
        ],
        "model__learning_rate": [
            0.1
        ]
    },
    "LightGBM": {
        "model__n_estimators": [
            100,
            200
        ],
        "model__num_leaves": [
            31,
            50
        ]
    },
    "CatBoost": {
        "model__iterations": [
            100,
            200
        ],
        "model__depth": [
            4,
            6
        ]
    }
}
# --------------------------------------------------
# Run Tuning
# --------------------------------------------------
if st.button("Run Hyperparameter Tuning"):
    pipe = Pipeline([
        ("preprocessor",
            preprocessor
        ),
        ("model",
            base_model
        )
    ])
    with st.spinner(
        "Running Grid Search..."
    ):
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grids[
                selected_model
            ],
            scoring="roc_auc",
            cv=3,
            n_jobs=1
        )
        try:
            grid.fit(
            X_train,
            y_train
        )
            st.session_state["tuned_model"] = grid.best_estimator_
            st.session_state["best_score"] = grid.best_score_
            st.session_state["best_params"] = grid.best_params_
            st.session_state["cv_results"] = grid.cv_results_
        except Exception as e:
            st.error(
            f"Tuning Failed: {e}")
            st.exception(e)
            st.stop()
       
        st.success(
            "Tuning Completed"
        )
# --------------------------------------------------
# Results
# --------------------------------------------------
if "tuned_model" in st.session_state:
    st.metric(
    "Best CV AUC",
    round(st.session_state["best_score"],4)
    )
    st.subheader(
        "Best Parameters"
    )
    st.json(
    st.session_state["best_params"])
    # --------------------------------------------------
    # Save Model Version
    # --------------------------------------------------
    os.makedirs(
        "models",
        exist_ok=True
    )
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    model_filename = (
        f"models/"
        f"{selected_model.replace(' ','_')}"
        f"_{timestamp}.pkl"
    )
    model_bytes = joblib.dump(
    st.session_state["tuned_model"],
    model_filename)
    
    st.success(
        f"Model Saved: {model_filename}"
    )
    # --------------------------------------------------
    # Download Model
    # --------------------------------------------------
    try:
        with open(model_filename,"rb") as f:
            st.download_button(
            "Download Tuned Model",
            f,
            file_name=model_filename.split("/")[-1]
        )
    except Exception as e:
        st.warning(
        f"Download unavailable: {e}"
    )
    # --------------------------------------------------
    # Registry
    # --------------------------------------------------
    registry_file = (
        "registry/model_registry.csv"
    )
    os.makedirs(
        "registry",
        exist_ok=True
    )
    registry_row = pd.DataFrame({
        "Timestamp": [
            timestamp
        ],
        "Model": [
            selected_model
        ],
        "CV_AUC": [
            st.session_state["best_score"]
        ],
        "File": [
            model_filename
        ]
    })
    if os.path.exists(
        registry_file
    ):
        old_registry = pd.read_csv(
            registry_file
        )
        registry = pd.concat(
            [
                old_registry,
                registry_row
            ],
            ignore_index=True
        )
    else:
        registry = registry_row
    try:
        registry.to_csv(
        registry_file,
        index=False
    )
    except Exception as e:
        st.warning(
        f"Registry could not be saved: {e}"
    )
    st.subheader(
        "Model Registry"
    )
    st.dataframe(
        registry
    )
    
    cv_results = pd.DataFrame(
    st.session_state["cv_results"]
)

    display_results = (
        cv_results[
            [
                "params",
                "mean_test_score",
                "std_test_score",
                "rank_test_score"
            ]
        ]
        .sort_values("rank_test_score")
        .head(20)
    )
    
    st.subheader(
        "Parameter Search Results"
    )
    st.dataframe(
        display_results,
        width="stretch"
    )

def calculate_psi(expected, actual, buckets=10):

    expected = np.array(expected)
    actual = np.array(actual)

    breakpoints = np.unique(
    np.percentile(
        expected,
        np.arange(0,buckets+1)/buckets*100
    ))
    
    if len(breakpoints) < 2:
        return 0

    expected_counts = np.histogram(
        expected,
        bins=breakpoints
    )[0]

    actual_counts = np.histogram(
        actual,
        bins=breakpoints
    )[0]

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    expected_pct = np.where(
        expected_pct == 0,
        0.0001,
        expected_pct
    )

    actual_pct = np.where(
        actual_pct == 0,
        0.0001,
        actual_pct
    )

    psi = np.sum(
        (actual_pct - expected_pct)
        *
        np.log(actual_pct / expected_pct)
    )

    return psi

st.divider()

st.header("Population Stability Index (PSI) Monitoring")

monitor_file = st.file_uploader(
    "Upload New Production Dataset",
    type=["csv"]
)

st.subheader(
    "Model Governance Summary"
)

if monitor_file:

    monitor_df = pd.read_csv(
        monitor_file
    )

    training_df = st.session_state[
        "training_data"
    ]

    num_cols = st.session_state[
        "num_cols"
    ]

    psi_results = []

    for col in num_cols:

        if col in monitor_df.columns:

            psi = calculate_psi(
                training_df[col].fillna(
                    training_df[col].median()
                ),
                monitor_df[col].fillna(
                    monitor_df[col].median()
                )
            )

            psi_results.append(
                [col, psi]
            )

    psi_df = pd.DataFrame(
        psi_results,
        columns=[
            "Feature",
            "PSI"
        ]
    )

    psi_df["Drift"] = psi_df[
        "PSI"
    ].apply(
        lambda x:
        "No Drift"
        if x < 0.1
        else
        "Moderate Drift"
        if x < 0.25
        else
        "Significant Drift"
    )

    
    st.subheader(
        "Feature-Level PSI"
    )

    st.dataframe(
        psi_df.sort_values(
            "PSI",
            ascending=False
        ),
        use_container_width=True
    )

    overall_psi = psi_df["PSI"].mean()
    
    col1,col2,col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Average PSI",
            round(overall_psi,4)
        )
    
    with col2:
        st.metric(
            "Max PSI",
            round(
                psi_df["PSI"].max(),
                4
            )
        )
    
    with col3:
    
        drifted_features = (
            psi_df["PSI"] > 0.25
        ).sum()
    
        st.metric(
            "Drifted Features",
            drifted_features
        )
    
    fig, ax = plt.subplots(
        figsize=(12,6)
    )
    
    psi_df.sort_values(
        "PSI",
        ascending=False
    ).head(20).plot(
        x="Feature",
        y="PSI",
        kind="bar",
        ax=ax
    )
    
    ax.axhline(
        y=0.1,
        linestyle="--",
        label="Moderate Drift"
    )
    
    ax.axhline(
        y=0.25,
        linestyle="--",
        label="Significant Drift"
    )
    
    ax.legend()
    
    st.pyplot(fig)
    
    train_scores = best_model.predict_proba(
        training_df
    )[:,1]

    expected_cols = training_df.columns.tolist()

    for col in expected_cols:
    
        if col not in monitor_df.columns:
    
            if col in st.session_state["num_cols"]:
                monitor_df[col] = 0
            else:
                monitor_df[col] = "Unknown"
    
    monitor_df = monitor_df[expected_cols]
    
    prod_scores = best_model.predict_proba(
        monitor_df[
            training_df.columns
        ]
    )[:,1]

    monitor_df[col] = pd.to_numeric(
    monitor_df[col],
    errors="coerce")
    
    score_psi = calculate_psi(
        train_scores,
        prod_scores
    )
    
    st.metric(
        "Score PSI",
        round(score_psi,4)
    )
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
cursor.execute("""
INSERT INTO registry
VALUES (?, ?, ?, ?)
""",
(
    timestamp,
    selected_model,
    grid.best_score_,
    str(st.session_state["best_params"])
))

conn.commit()

registry = pd.read_sql(
    "SELECT * FROM registry",
    conn
)

st.dataframe(
    registry,
    use_container_width=True
)
