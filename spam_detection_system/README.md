# Spam Detection System

This tool classifies emails as spam or non-spam using a simple KNN model.

## Features
- Preprocesses email data (CSV format)
- Applies TF-IDF text feature extraction
- Trains a KNN classifier
- Evaluates accuracy using a confusion matrix
- Provides a user-friendly web interface for predictions

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- streamlit

Install dependencies with:
```
pip install -r requirements.txt
```

## Usage
1. Place your email CSV file in the project folder (or use the provided sample).
2. Run the app:
```
streamlit run spam_detection_system.py
```

The web interface will open in your browser for predictions. 