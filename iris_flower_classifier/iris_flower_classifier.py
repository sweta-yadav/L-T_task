import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Iris Flower Classifier')

# Load Iris dataset
data = load_iris(as_frame=True)
df = data.frame
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate
preds = knn.predict(X_test)
acc = accuracy_score(y_test, preds)
cm = confusion_matrix(y_test, preds)

st.write(f'### Model Accuracy: {acc:.2f}')
st.write('### Confusion Matrix:')
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names, ax=ax)
st.pyplot(fig)

st.write('---')
st.write('## Predict Iris Species')
def user_input_features():
    sepal_length = st.number_input('Sepal length (cm)', min_value=4.0, max_value=8.0, value=5.1)
    sepal_width = st.number_input('Sepal width (cm)', min_value=2.0, max_value=5.0, value=3.5)
    petal_length = st.number_input('Petal length (cm)', min_value=1.0, max_value=7.0, value=1.4)
    petal_width = st.number_input('Petal width (cm)', min_value=0.1, max_value=3.0, value=0.2)
    data_dict = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    return pd.DataFrame([data_dict])

input_df = user_input_features()
if st.button('Predict'):
    prediction = knn.predict(input_df)[0]
    st.success(f'Predicted Iris Species: {data.target_names[prediction]}') 