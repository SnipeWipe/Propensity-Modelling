# Customer Acquisition Risk & Propensity Modeling Platform

## App Link - https://propensity-modelling-aypi2m9hwzdvcwdhyf48eb.streamlit.app/

## Overview

The Customer Acquisition Risk & Propensity Modeling Platform is an end-to-end Machine Learning application built using Streamlit for customer propensity prediction, model explainability, monitoring, governance, and business decision support.

The platform enables users to:

* Analyze data quality
* Perform WOE & IV analysis
* Train multiple machine learning models
* Compare model performance
* Explain predictions using SHAP
* Score new customers
* Monitor model drift using PSI
* Perform hyperparameter tuning
* Maintain model governance and registry

---

## Architecture

```text
Upload Dataset
       │
       ▼
Data Quality Dashboard
       │
       ▼
WOE & IV Analysis
       │
       ▼
Model Training
       │
       ▼
Model Explainability
       │
       ▼
Customer Scoring
       │
       ▼
Model Monitoring & Drift Detection
       │
       ▼
Business Insights
```

---

## Features

### 1. Data Quality Dashboard

* Duplicate detection and removal
* Missing value analysis
* Dataset profiling
* Statistical summary
* Correlation heatmap
* Feature distribution analysis
* Outlier detection
* Outlier treatment:

  * IQR Capping
  * Percentile Capping
  * Log Transformation
  * Outlier Removal
* Categorical feature analysis

---

### 2. WOE & IV Analysis

Performs Weight of Evidence (WOE) and Information Value (IV) calculations for feature selection.

#### IV Interpretation

| IV Value    | Predictive Power      |
| ----------- | --------------------- |
| < 0.02      | Not Useful            |
| 0.02 – 0.10 | Weak                  |
| 0.10 – 0.30 | Medium                |
| 0.30 – 0.50 | Strong                |
| > 0.50      | Suspiciously Powerful |

#### Outputs

* IV Report
* WOE Tables
* Interactive WOE Visualization
* Downloadable IV Report

---

### 3. Model Training

The platform automatically trains multiple classification algorithms.

#### Supported Models

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM
* CatBoost

#### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* GINI
* KS Statistic

#### Visualizations

* ROC Curves
* Precision-Recall Curves
* Confusion Matrix
* Lift Chart
* Decile Analysis

#### Best Model Selection

The best model is selected automatically based on the highest ROC-AUC score.

---

### 4. Model Explainability

Provides model interpretability using SHAP.

#### Global Explainability

* SHAP Summary Plot
* SHAP Feature Importance Plot

#### Local Explainability

* Customer-Level SHAP Waterfall Plot
* Top Positive Drivers
* Top Negative Drivers

This allows business users to understand why a customer receives a specific propensity score.

---

### 5. Customer Scoring

Score new customer datasets using the trained model.

#### Features

* Schema validation
* Missing column handling
* Extra column detection
* Automatic preprocessing
* Customer segmentation

#### Segments

| Score Range | Segment |
| ----------- | ------- |
| 0.00 – 0.30 | Low     |
| 0.30 – 0.60 | Medium  |
| 0.60 – 1.00 | High    |

#### Outputs

* Propensity Scores
* Customer Segments
* Top 100 High Propensity Customers
* Downloadable Prediction File

---

### 6. Hyperparameter Tuning

Optimize model performance using GridSearchCV.

#### Supported Models

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM
* CatBoost

#### Outputs

* Best Parameters
* Best Cross Validation AUC
* Tuned Model Export
* Model Registry Tracking

---

### 7. Model Monitoring

Monitor production model performance and data drift.

#### PSI (Population Stability Index)

Measures feature distribution shifts between training and production datasets.

#### Drift Interpretation

| PSI         | Interpretation    |
| ----------- | ----------------- |
| < 0.10      | No Drift          |
| 0.10 – 0.25 | Moderate Drift    |
| > 0.25      | Significant Drift |

#### Monitoring Outputs

* Feature-level PSI
* Score PSI
* Drift Summary
* Governance Dashboard

---

### 8. Business Insights

Provides executive-level model performance metrics.

#### KPIs

* Best AUC
* Best KS
* Best GINI

Useful for business stakeholders and model governance reviews.

---

## Project Structure

```text
Customer-Acquisition-Propensity-Modeling/

│
├── app.py
│
├── pages/
│   ├── data_quality.py
│   ├── woe_iv.py
│   ├── training.py
│   ├── model_explainability.py
│   ├── customer_scoring.py
│   ├── model_monitoring.py
│   └── business_insights.py
│
├── models/
│   ├── best_model.pkl
│
├── registry/
│   └── model_registry.csv
│
├── model_registry.db
│
├── requirements.txt
│
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/customer-propensity-modeling.git

cd customer-propensity-modeling
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Mac/Linux

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
streamlit run app.py
```

---

## Required Libraries

```text
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
catboost
shap
joblib
altair
sqlite3
```

---

## Sample Workflow

### Step 1

Upload customer dataset.

### Step 2

Perform data quality checks.

### Step 3

Run WOE & IV analysis.

### Step 4

Train machine learning models.

### Step 5

Review model comparison metrics.

### Step 6

Analyze SHAP explainability.

### Step 7

Score new customers.

### Step 8

Monitor production drift using PSI.

### Step 9

Tune hyperparameters for better performance.

### Step 10

Track models through governance registry.

---

## Business Applications

### Banking

* Term Deposit Campaign Optimization
* Cross-Sell Prediction
* Loan Offer Acceptance Prediction

### Insurance

* Policy Purchase Propensity
* Customer Retention

### FinTech

* Lead Prioritization
* Customer Conversion Prediction

### Marketing

* Campaign Targeting
* Customer Segmentation

---

## Future Enhancements

* MLflow Integration
* Model Versioning API
* Real-Time Scoring API
* Automated Retraining Pipeline
* Feature Store Integration
* Docker Deployment
* AWS/Azure Deployment
* Champion-Challenger Framework

---

## Author

**Akhilesh Pal**

Data Science | Machine Learning | Credit Risk Analytics | MLOps

---

## License

This project is licensed under the MIT License.
