import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Data Quality Dashboard")

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Uploaded Successfully")

    # Remove duplicates
    duplicates = df.duplicated().sum()

    if duplicates > 0:
        df = df.drop_duplicates()

    st.metric("Duplicate Records Removed", duplicates)

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
