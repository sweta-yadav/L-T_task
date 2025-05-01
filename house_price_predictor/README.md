# House Price Predictor

This tool predicts house prices using a simple linear regression model.

## Features
- Preprocesses the housing dataset (CSV format)
- Trains a linear regression model
- Evaluates the model using mean squared error
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
1. Place your housing CSV file in the project folder (or use the provided sample).
2. Run the app:
```
streamlit run house_price_predictor.py
```

The web interface will open in your browser for predictions. 