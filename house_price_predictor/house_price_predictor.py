import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

st.title('House Price Predictor')

# Load dataset
def load_data(file):
    return pd.read_csv(file)

# Preprocess dataset
def preprocess(df):
    X = df.drop('price', axis=1)
    y = df['price']
    return X, y

# Train model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Main app
uploaded_file = st.file_uploader('Upload your housing CSV file', type=['csv'])
if uploaded_file:
    df = load_data(uploaded_file)
else:
    df = load_data('sample_housing.csv')

st.write('### Dataset Preview')
st.dataframe(df.head())

X, y = preprocess(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = train_model(X_train, y_train)

# Evaluate model
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
st.write(f'### Model Mean Squared Error: {mse:.2f}')

st.write('---')
st.write('## Predict House Price')

def user_input_features():
    area = st.number_input('Area (sq ft)', min_value=500, max_value=10000, value=2000)
    bedrooms = st.number_input('Bedrooms', min_value=1, max_value=10, value=3)
    age = st.number_input('Age of House (years)', min_value=0, max_value=100, value=10)
    data = {'area': area, 'bedrooms': bedrooms, 'age': age}
    return pd.DataFrame([data])

input_df = user_input_features()

if st.button('Predict'):
    prediction = model.predict(input_df)[0]
    st.success(f'Estimated House Price: ${prediction:,.2f}') 