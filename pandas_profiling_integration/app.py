import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit.components.v1 import html
import tempfile

st.set_page_config(page_title="Pandas Profiling Integration", layout="wide")
st.title("Pandas Profiling EDA App")

st.write("""
Upload a CSV file to generate an interactive Exploratory Data Analysis (EDA) report using Pandas Profiling.
""")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    st.subheader("Key Metrics")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write(f"Missing values: {df.isnull().sum().sum()}")
    st.write("Correlation matrix:")
    st.dataframe(df.corr(numeric_only=True))

    if st.button("Generate EDA Report"):
        with st.spinner("Generating report..."):
            profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_file:
                profile.to_file(tmp_file.name)
                with open(tmp_file.name, "r", encoding="utf-8") as f:
                    report_html = f.read()
                html(report_html, height=1000, scrolling=True)
            st.success("EDA report generated!")
else:
    st.info("Please upload a CSV file to begin.") 