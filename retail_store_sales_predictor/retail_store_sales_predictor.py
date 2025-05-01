import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

st.title('Retail Store Sales Predictor')

def load_data(file):
    return pd.read_csv(file)

uploaded_file = st.file_uploader('Upload your sales CSV file', type=['csv'])
if uploaded_file:
    df = load_data(uploaded_file)
else:
    df = load_data('sample_sales.csv')

st.write('### Dataset Preview')
st.dataframe(df.head())

# Preprocessing
X = df.drop('sales', axis=1)
y = df['sales']

# Column transformer for encoding and scaling
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(), ['location']),
    ('num', StandardScaler(), ['ad_budget'])
])
X_processed = preprocessor.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
st.write(f'### Model Mean Squared Error: {mse:.2f}')

st.write('---')
st.write('## Predict Sales')
def user_input_features():
    location = st.selectbox('Location', sorted(df["location"].unique()))
    ad_budget = st.number_input('Advertisement Budget', min_value=0, value=10000)
    data = {'location': location, 'ad_budget': ad_budget}
    return pd.DataFrame([data])

input_df = user_input_features()
input_processed = preprocessor.transform(input_df)
if st.button('Predict'):
    prediction = model.predict(input_processed)[0]
    st.success(f'Estimated Sales: {prediction:,.2f}') 