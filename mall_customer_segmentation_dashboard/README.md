# Mall Customer Segmentation Dashboard

This tool categorizes mall customer types using spending patterns and visualizes clusters with K-means.

## Features
- Loads mall customer data (CSV format)
- Performs feature scaling
- Applies K-means clustering
- Generates and presents visualizations
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
1. Place your mall customer CSV file in the project folder (or use the provided sample).
2. Run the app:
```
streamlit run mall_customer_segmentation_dashboard.py
```

The web interface will open in your browser for exploration. 