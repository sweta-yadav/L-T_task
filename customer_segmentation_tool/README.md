# Customer Segmentation Tool

This tool groups customers using K-means clustering for targeted marketing.

## Features
- Loads customer transaction data (CSV format)
- Applies feature scaling
- Trains a K-means clustering model
- Visualizes customer clusters
- Provides a user-friendly web interface for exploration

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- streamlit
- matplotlib
- seaborn

Install dependencies with:
```
pip install -r requirements.txt
```

## Usage
1. Place your customer CSV file in the project folder (or use the provided sample).
2. Run the app:
```
streamlit run customer_segmentation_tool.py
```

The web interface will open in your browser for exploration. 