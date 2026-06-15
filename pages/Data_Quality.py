import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Data Quality Dashboard")

# Check if dataset is loaded
if "df" not in st.session_state:
    st.warning("Please upload a dataset from the Home page.")
    st.stop()

df = st.session_state["df"]

# Remove duplicates
duplicates = df.duplicated().sum()

if duplicates > 0:
    df = df.drop_duplicates()

st.metric(
    "Duplicate Records Removed",
    duplicates
)

# Save cleaned dataframe back
st.session_state["df"] = df

# Display data
st.dataframe(
    df,
    height=500,
    use_container_width=True
)

# Data types
st.subheader("Data Types")

dtype_df = pd.DataFrame({
    "Column": df.columns,
    "Datatype": df.dtypes.astype(str).values
})

st.dataframe(
    dtype_df,
    use_container_width=True
)

st.subheader("Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Rows",
        df.shape[0])
with c2:
    st.metric(
        "Columns",
        df.shape[1])
with c3:
    st.metric(
        "Numeric Features",
        len(
            df.select_dtypes(
                include="number"
            ).columns))
with c4:
    st.metric(
        "Categorical Features",
        len(
            df.select_dtypes(
                exclude="number"
            ).columns))

st.subheader("Statistical Summary")

st.dataframe(
    df.describe().T,
    use_container_width=True
)

if "deposit" in df.columns:
    st.subheader(
        "Target Distribution"
    )
    fig, ax = plt.subplots()
    df["deposit"].value_counts().plot(
        kind="bar",
        ax=ax
    )
    ax.set_xlabel("Target")
    ax.set_ylabel("Count")
    st.pyplot(fig)

numeric_cols = df.select_dtypes(
    include="number"
)

if len(numeric_cols.columns) > 1:
    st.subheader(
        "Correlation Heatmap"
    )
    fig, ax = plt.subplots(
        figsize=(10,6)
    )
    corr_df = numeric_cols.drop(
    columns=["deposit"],
    errors="ignore"
)
    sns.heatmap(
        corr_df.corr(),
        cmap="coolwarm",
        ax=ax
    )
    st.pyplot(fig)

st.subheader(
    "Feature Distribution"
)
num_cols = df.select_dtypes(
    include="number").columns

if len(num_cols) > 0:
    selected_feature = st.selectbox(
        "Select Numeric Feature",
        num_cols
    )
    fig, ax = plt.subplots()
    df[selected_feature].hist(
        bins=30,
        ax=ax
    )
    st.pyplot(fig)
else:
    st.warning("No numeric columns found.")

st.subheader(
    "Outlier Detection"
)
if len(num_cols) > 0:
    box_feature = st.selectbox(
        "Select Feature for Box Plot",
        num_cols,
        key="box"
    )

    if st.button("Apply Treatment"):

    treated_df = df.copy()

    Q1 = treated_df[box_feature].quantile(0.25)
    Q3 = treated_df[box_feature].quantile(0.75)

    IQR = Q3 - Q1

     if IQR == 0:
    st.warning(
        "Selected feature has no variation."
    )

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    if treatment == "IQR Capping":

        treated_df[box_feature] = (
            treated_df[box_feature]
            .clip(lower, upper)
        )

        st.success(
            f"IQR Capping applied to {box_feature}"
        )

    elif treatment == "Percentile Capping (1%-99%)":

        lower = treated_df[
            box_feature
        ].quantile(0.01)

        upper = treated_df[
            box_feature
        ].quantile(0.99)

        treated_df[box_feature] = (
            treated_df[box_feature]
            .clip(lower, upper)
        )

        st.success(
            f"Percentile Capping applied to {box_feature}"
        )

    elif treatment == "Log Transformation":

        if (treated_df[box_feature] < 0).any():
            st.error(
                "Log transformation cannot be applied to negative values."
            )
        else:
            treated_df[box_feature] = np.log1p(
                treated_df[box_feature])

            st.success(
                f"Log Transformation applied to {box_feature}"
            )

    elif treatment == "Remove Outliers":

        treated_df = treated_df[
            (
                treated_df[box_feature]
                >= lower
            )
            &
            (
                treated_df[box_feature]
                <= upper
            )
        ]

        st.success(
            f"Outliers removed from {box_feature}"
        )

    st.session_state["df"] = treated_df
    df = treated_df
    
    st.write(
        "Updated Dataset Shape:",
        treated_df.shape
    )

    st.dataframe(
        treated_df.head(),
        use_container_width=True
    )

    fig, ax = plt.subplots()
    ax.boxplot(
        df[box_feature].dropna()
    )
    ax.set_title(box_feature)
    st.pyplot(fig)
else:
    st.warning("No numeric columns available for outlier analysis.")

cat_cols = df.select_dtypes(
    exclude="number"
).columns
if len(cat_cols) > 0:
    st.subheader(
        "Categorical Analysis"
    )
    cat_feature = st.selectbox(
        "Select Categorical Feature",
        cat_cols
    )
    fig, ax = plt.subplots(
        figsize=(10,5)
    )
    df[cat_feature].value_counts().head(
        15
    ).plot(
        kind="bar",
        ax=ax
    )
    st.pyplot(fig)

st.subheader("Outlier Treatment")

treatment = st.selectbox(
    "Select Treatment Method",
    [
        "None",
        "IQR Capping",
        "Percentile Capping (1%-99%)",
        "Log Transformation",
        "Remove Outliers"
    ]
)
    
missing_df = pd.DataFrame({
    "Column": df.columns,
    "Missing Count": df.isnull().sum().values,
    "Missing %": round(
        (df.isnull().sum() / len(df)) * 100,
        2
    ).values
})
st.subheader("Missing Values")
st.dataframe(
    missing_df,
    use_container_width=True
)
if missing_df["Missing Count"].sum() > 0:
    st.subheader(
        "Missing Value Distribution"
    )
    fig, ax = plt.subplots(
        figsize=(10,5)
    )
    missing_df.sort_values(
        "Missing %",
        ascending=False
    ).plot(
        x="Column",
        y="Missing %",
        kind="bar",
        ax=ax
    )
    st.pyplot(fig)
else:
    st.success(
        "No missing values detected."
    )
    
st.subheader(
    "Feature Monitoring Summary"
)
feature_summary = pd.DataFrame()
feature_summary["Feature"] = df.columns
feature_summary["Unique Values"] = [
    df[col].nunique()
    for col in df.columns
]
st.dataframe(
    feature_summary,
    use_container_width=True
)
st.session_state["num_cols"] = (
    df.select_dtypes(
        include="number"
    ).columns.tolist()
)
st.session_state["cat_cols"] = (
    df.select_dtypes(
        exclude="number"
    ).columns.tolist()
)
st.session_state["training_data"] = df.copy()


