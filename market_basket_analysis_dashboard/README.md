# Market Basket Analysis Dashboard

This dashboard visualizes customer buying patterns and correlation analysis using association rule mining.

## Features
- Loads transactional sales data (CSV format)
- Performs association rule mining
- Generates visual insights using heatmaps
- Highlights frequently purchased item sets
- Provides a user-friendly web interface for exploration

## Requirements
- Python 3.x
- pandas
- numpy
- mlxtend
- streamlit
- matplotlib
- seaborn

Install dependencies with:
```
pip install -r requirements.txt
```

## Usage
1. Place your transactional sales CSV file in the project folder (or use the provided sample).
2. Run the app:
```
streamlit run market_basket_analysis_dashboard.py
```

The web interface will open in your browser for exploration. 