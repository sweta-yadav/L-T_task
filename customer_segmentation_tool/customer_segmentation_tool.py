import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Customer Segmentation Tool')

def load_data(file):
    return pd.read_csv(file)

uploaded_file = st.file_uploader('Upload your customer CSV file', type=['csv'])
if uploaded_file:
    df = load_data(uploaded_file)
else:
    df = load_data('sample_customers.csv')

st.write('### Dataset Preview')
st.dataframe(df.head())

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# K-means clustering
k = st.slider('Select number of clusters (k)', min_value=2, max_value=8, value=3)
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

st.write('### Clustered Data Preview')
st.dataframe(df.head())

# Visualize clusters
st.write('### Cluster Visualization')
fig, ax = plt.subplots()
sns.scatterplot(x=df.columns[0], y=df.columns[1], hue='Cluster', palette='Set2', data=df, ax=ax, s=80)
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
st.pyplot(fig) 