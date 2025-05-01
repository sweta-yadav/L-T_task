# E-commerce Data Cleaner

A Python tool for cleaning and preprocessing e-commerce datasets to prepare them for machine learning models.

## Features

- Import raw e-commerce data from CSV files
- Detect and handle missing entries using appropriate strategies
- Identify outliers in price and quantity columns
- Generate detailed cleaning reports and visualizations
- Save cleaned data in a structured format

## Requirements

- Python 3.7 or higher
- Required packages listed in `requirements.txt`

## Installation

1. Clone or download this repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your e-commerce data CSV file in the project directory
2. Rename it to `sample_ecommerce_data.csv` or update the filename in `ecommerce_data_cleaner.py`
3. Run the script:
```bash
python ecommerce_data_cleaner.py
```

## Input Data Format

The cleaner expects a CSV file with the following columns (at minimum):
- Product information (ID, name, category, etc.)
- Price
- Quantity
- Order/transaction details
- Customer information

## Output

The script generates:
1. Cleaned dataset (`cleaned_ecommerce_data.csv`)
2. Detailed cleaning report (`cleaning_report.txt`)
3. Outlier visualization plots in the `outlier_plots` directory

## Data Cleaning Process

1. **Loading Data**
   - Imports CSV file
   - Validates data structure

2. **Missing Value Treatment**
   - Detects missing entries
   - Applies appropriate filling strategies:
     - Numeric columns: median
     - Categorical columns: mode

3. **Outlier Detection**
   - Uses Z-score method
   - Focuses on price and quantity columns
   - Generates before/after visualization plots

4. **Data Export**
   - Saves cleaned dataset
   - Generates comprehensive cleaning report
   - Creates visualization plots

## Example

The repository includes a sample dataset (`sample_ecommerce_data.csv`) with common data quality issues for testing purposes. 