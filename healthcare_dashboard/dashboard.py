import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Load patient diagnostic data
data = pd.read_csv('patient_data.csv')

st.title("Healthcare Data Dashboard")

# Show data
st.subheader("Patient Data")
st.write(data)

# Generate scatter plots for feature relationships
st.subheader("Scatter Plot: Age vs Blood Pressure")
fig1, ax1 = plt.subplots()
sns.scatterplot(data=data, x='Age', y='BloodPressure', hue='Outcome', ax=ax1)
st.pyplot(fig1)

# Analyze correlations between symptoms and outcomes
st.subheader("Correlation Heatmap")
corr = data.drop(['PatientID', 'Outcome'], axis=1).corr()
fig2, ax2 = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax2)
st.pyplot(fig2)

# Present insights using dashboards
st.subheader("Outcome Distribution")
st.bar_chart(data['Outcome'].value_counts()) 