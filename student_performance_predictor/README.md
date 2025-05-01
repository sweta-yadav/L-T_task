# Student Performance Predictor

This tool predicts student exam scores based on attendance and assignment scores using regression.

## Features
- Loads an academic dataset (CSV format)
- Performs feature scaling and encoding
- Trains a regression model
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
1. Place your academic CSV file in the project folder (or use the provided sample).
2. Run the app:
```
streamlit run student_performance_predictor.py
```

The web interface will open in your browser for predictions. 