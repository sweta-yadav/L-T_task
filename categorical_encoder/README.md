# Categorical Data Encoder

A Python tool for automating the process of encoding categorical data for machine learning models. This tool supports both label encoding and one-hot encoding methods.

## Features

- Automatic detection of categorical columns in datasets
- Support for both label encoding and one-hot encoding
- Detailed encoding reports with mappings
- Easy-to-use command-line interface
- Preserves non-categorical columns

## Requirements

- Python 3.7 or higher
- Required packages (install using `pip install -r requirements.txt`):
  - pandas
  - numpy
  - scikit-learn

## Installation

1. Clone or download this repository
2. Navigate to the project directory
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your CSV file in an accessible location
2. Run the script:
   ```bash
   python categorical_encoder.py
   ```
3. Follow the interactive prompts:
   - Enter the path to your CSV file
   - Review detected categorical columns
   - Choose encoding method (label or one-hot encoding)
   - Get encoded data and report files

## Input Data Format

The tool accepts CSV files with any number of columns. It automatically detects categorical columns (text or categorical data types).

## Output Files

1. Encoded dataset: `[original_filename]_encoded.csv`
   - Contains the transformed dataset with encoded categorical columns
   - Original non-categorical columns are preserved

2. Encoding report: `encoding_report.txt`
   - Lists all detected categorical columns
   - Shows number of unique values per column
   - Provides encoding mappings (for label encoding)
   - Records any errors or issues during encoding

## Encoding Methods

1. **Label Encoding**
   - Converts categorical values to numeric labels (0 to n-1)
   - Maintains a single column per category
   - Best for ordinal categorical data

2. **One-Hot Encoding**
   - Creates binary columns for each unique category
   - Increases dimensionality but avoids ordinal relationships
   - Best for nominal categorical data

## Example

```python
# Sample usage in Python
from categorical_encoder import CategoricalDataEncoder

encoder = CategoricalDataEncoder()
encoder.load_data("sample_data.csv")
encoder.identify_categorical_columns()
encoder.apply_label_encoding()  # or encoder.apply_onehot_encoding()
encoder.save_encoded_data("encoded_data.csv")
encoder.save_report()
```

## Notes

- The tool automatically handles missing values in categorical columns
- One-hot encoding may significantly increase the number of columns for categories with many unique values
- Label encoding preserves memory but may introduce unwanted ordinal relationships 