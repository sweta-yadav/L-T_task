import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Crime Rate Visualization Tool", layout="wide")
st.title("Crime Rate Visualization Tool")

st.write("""
A visualization platform to analyze crime trends and correlations. Upload your crime rate data or use the sample provided.
""")

uploaded_file = st.file_uploader("Upload your crime rate CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using sample data: crime_data_sample.csv")
    df = pd.read_csv("crime_rate_visualization_tool/crime_data_sample.csv")

st.subheader("Data Preview")
st.dataframe(df.head())

# Select columns for plotting
date_col = st.selectbox("Select date column", options=df.columns, index=0)
value_col = st.selectbox("Select value column (crime rate)", options=df.columns, index=1)
category_col = st.selectbox("Select category column (if any)", options=[None] + list(df.columns), index=0)

# Ensure value column is numeric
try:
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
except Exception as e:
    st.error(f"Could not convert {value_col} to numeric: {e}")

# Show dtypes for debugging
st.write("Column types:", df.dtypes)

# Only plot if value_col is numeric and has data
if df[value_col].dropna().empty or not pd.api.types.is_numeric_dtype(df[value_col]):
    st.error("Selected value column does not contain numeric data to plot.")
else:
    st.subheader("Line Plot: Crime Rate Over Time")
    fig, ax = plt.subplots()
    if category_col and category_col in df.columns:
        for cat in df[category_col].unique():
            sub_df = df[df[category_col] == cat]
            ax.plot(sub_df[date_col], sub_df[value_col], label=str(cat))
        ax.legend()
    else:
        ax.plot(df[date_col], df[value_col])
    ax.set_xlabel(date_col)
    ax.set_ylabel(value_col)
    ax.set_title("Crime Rate Over Time")
    st.pyplot(fig)

    st.subheader("Bar Plot: Crime Rate by Category/Time")
    fig2, ax2 = plt.subplots()
    if category_col and category_col in df.columns:
        grouped = df.groupby(category_col)[value_col].sum()
        # Only plot if grouped has numeric data
        if grouped.dropna().empty or not pd.api.types.is_numeric_dtype(grouped):
            st.error("No numeric data to plot for the selected category and value columns.")
        else:
            grouped.plot(kind="bar", ax=ax2)
            ax2.set_xlabel(category_col)
            ax2.set_ylabel(value_col)
            ax2.set_title("Crime Rate Bar Plot")
            st.pyplot(fig2)
    else:
        grouped = df.groupby(date_col)[value_col].sum()
        if grouped.dropna().empty or not pd.api.types.is_numeric_dtype(grouped):
            st.error("No numeric data to plot for the selected date and value columns.")
        else:
            grouped.plot(kind="bar", ax=ax2)
            ax2.set_xlabel(date_col)
            ax2.set_ylabel(value_col)
            ax2.set_title("Crime Rate Bar Plot")
            st.pyplot(fig2)

    st.subheader("Pattern & Anomaly Detection")
    mean_val = df[value_col].mean()
    std_val = df[value_col].std()
    anomalies = df[df[value_col] > mean_val + 2*std_val]
    if not anomalies.empty:
        st.warning(f"Anomalies detected (values > mean + 2*std):")
        st.dataframe(anomalies)
    else:
        st.success("No significant anomalies detected.") 