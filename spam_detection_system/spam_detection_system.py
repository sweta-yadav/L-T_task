import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

st.title('Spam Detection System')

def load_data(file):
    return pd.read_csv(file)

def preprocess(df):
    X = df['text']
    # Clean label: strip spaces, lowercase, map, and drop NaNs
    y = df['label'].astype(str).str.strip().str.lower().map({'ham': 0, 'spam': 1})
    mask = y.notna()
    return X[mask], y[mask]

def train_knn(X_train_vec, y_train):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_vec, y_train)
    return knn

uploaded_file = st.file_uploader('Upload your email CSV file', type=['csv'])
if uploaded_file:
    df = load_data(uploaded_file)
else:
    df = load_data('sample_emails.csv')

st.write('### Dataset Preview')
st.dataframe(df.head())

X, y = preprocess(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

knn = train_knn(X_train_vec, y_train)
preds = knn.predict(X_test_vec)
acc = accuracy_score(y_test, preds)
cm = confusion_matrix(y_test, preds)

st.write(f'### Model Accuracy: {acc:.2f}')
st.write('### Confusion Matrix:')
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=ax)
st.pyplot(fig)

st.write('---')
st.write('## Predict Email Spam or Not')
user_input = st.text_area('Enter email text:')
if st.button('Predict') and user_input:
    user_vec = vectorizer.transform([user_input])
    pred = knn.predict(user_vec)[0]
    label = 'Spam' if pred == 1 else 'Ham'
    st.success(f'This email is predicted as: {label}') 