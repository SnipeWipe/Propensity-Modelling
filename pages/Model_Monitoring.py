import streamlit as st
import pandas as pd
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
            100,
            200
        ],
        "model__max_depth": [
            3,
            5,
            7
        ],
        "model__learning_rate": [
            0.01,
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
            200,
            500
        ],
        "model__depth": [
            4,
            6,
            8
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
            cv=5,
            n_jobs=-1
        )
        grid.fit(
            X_train,
            y_train
        )
        st.session_state[
            "tuned_grid"
        ] = grid
        st.success(
            "Tuning Completed"
        )
# --------------------------------------------------
# Results
# --------------------------------------------------
if "tuned_grid" in st.session_state:
    grid = st.session_state[
        "tuned_grid"
    ]
    st.metric(
        "Best CV AUC",
        round(
            grid.best_score_,
            4)
    )
    st.subheader(
        "Best Parameters"
    )
    st.json(
        grid.best_params_
    )
    st.session_state[
        "tuned_model"
    ] = grid.best_estimator_
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
    joblib.dump(
        grid.best_estimator_,
        model_filename
    )
    st.success(
        f"Model Saved: {model_filename}"
    )
    # --------------------------------------------------
    # Download Model
    # --------------------------------------------------
    with open(
        model_filename,
        "rb"
    ) as f:
        st.download_button(
            "Download Tuned Model",
            f,
            file_name=model_filename.split("/")[-1]
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
            grid.best_score_
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
    registry.to_csv(
        registry_file,
        index=False
    )
    st.subheader(
        "Model Registry"
    )
    st.dataframe(
        registry
    )
    
    cv_results = pd.DataFrame(
        grid.cv_results_
    )
    st.subheader(
        "Parameter Search Results"
    )
    st.dataframe(
        cv_results[
            [
                "params",
                "mean_test_score",
                "std_test_score",
                "rank_test_score"
            ]
        ].sort_values(
            "rank_test_score"
        )
    )