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
    df = df.drop("duration",axis=1,inplace=True)
    st.session_state["df"] = df

    st.success("Dataset Uploaded Successfully")

    st.dataframe(
        df,
        height=500,
        use_container_width=True
    )


    st.subheader("Missing Values")

    missing_df = pd.DataFrame({
    "Column": df.columns,
    "Missing %" : (
        df.isnull().sum()
        /
        len(df)
        *
        100
    )
})

    st.dataframe(missing_df)

    duplicates = df.duplicated().sum()

    st.metric(
    "Duplicate Records",
    duplicates
)

    st.write("Data Types")
    dtype_df = pd.DataFrame({
    "Column": df.columns,
    "Datatype":
    df.dtypes.astype(str)
})

    st.dataframe(dtype_df)
