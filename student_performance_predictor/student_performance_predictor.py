import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.title('Student Performance Predictor')

# Load dataset
def load_data(file):
    return pd.read_csv(file)

# Preprocess dataset
def preprocess(df):
    X = df.drop('exam_score', axis=1)
    y = df['exam_score']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

# Train model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Main app
uploaded_file = st.file_uploader('Upload your academic CSV file', type=['csv'])
if uploaded_file:
    df = load_data(uploaded_file)
else:
    df = load_data('sample_academic.csv')

st.write('### Dataset Preview')
st.dataframe(df.head())

X, y, scaler = preprocess(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = train_model(X_train, y_train)

# Evaluate model
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
st.write(f'### Model Mean Squared Error: {mse:.2f}')

st.write('---')
st.write('## Predict Student Exam Score')

def user_input_features():
    attendance = st.number_input('Attendance (%)', min_value=0, max_value=100, value=85)
    assignments = st.number_input('Assignment Score (%)', min_value=0, max_value=100, value=80)
    data = {'attendance': attendance, 'assignments': assignments}
    return pd.DataFrame([data])

input_df = user_input_features()
input_scaled = scaler.transform(input_df)

if st.button('Predict'):
    prediction = model.predict(input_scaled)[0]
    st.success(f'Estimated Exam Score: {prediction:.2f}') 