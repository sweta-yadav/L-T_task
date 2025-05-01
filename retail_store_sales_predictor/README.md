# Retail Store Sales Predictor

This tool forecasts sales based on factors like location and advertisement budget using linear regression.

## Features
- Loads sales dataset (CSV format)
- Preprocesses data by scaling and encoding
- Trains a linear regression model
- Evaluates and deploys the model
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
1. Place your sales CSV file in the project folder (or use the provided sample).
2. Run the app:
```
streamlit run retail_store_sales_predictor.py
```

The web interface will open in your browser for predictions. 