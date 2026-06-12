import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Data Quality Dashboard")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # Drop duration column if present
    if "duration" in df.columns:
        df = df.drop("duration", axis=1)

    st.success("Dataset Uploaded Successfully")

    # Remove duplicates
    duplicates = df.duplicated().sum()

    if duplicates > 0:
        df = df.drop_duplicates()

    st.metric("Duplicate Records Removed", duplicates)

    # Convert target variable
    if "deposit" in df.columns:
        df["deposit"] = df["deposit"].map({
            "no": 0,
            "yes": 1
        })

    # Save cleaned dataframe
    st.session_state["df"] = df

    # Display data
    st.dataframe(
        df,
        height=500,
        use_container_width=True
    )

    # Missing values
    st.subheader("Missing Values")

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isnull().sum().values,
        "Missing %": round(
            (df.isnull().sum() / len(df)) * 100,
            2
        ).values
    })

    st.dataframe(
        missing_df,
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
