# Data Split Automation Tool

A Python tool for automating the process of splitting datasets into training, testing, and validation subsets while ensuring balanced class distribution.

## Features

- Automated splitting of datasets into train/test/validation sets
- Support for CSV and Excel file formats
- Stratified splitting to maintain class distribution
- Class distribution analysis and visualization
- Configurable via YAML configuration file
- Command-line interface for easy use
- Detailed logging of the splitting process

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Basic usage:
```bash
python data_splitter.py --input data.csv --target target_column --output-dir splits
```

All options:
```bash
python data_splitter.py \
    --input data.csv \
    --target target_column \
    --output-dir splits \
    --config config.yaml \
    --test-size 0.2 \
    --val-size 0.2 \
    --no-stratify \
    --random-state 42 \
    --prefix experiment1_
```

### Python API

```python
from data_splitter import DataSplitter

# Initialize splitter
splitter = DataSplitter(config_path='config.yaml')

# Load data
X, y = splitter.load_data('data.csv', target_column='target')

# Analyze class distribution
stats = splitter.analyze_class_distribution(y)
splitter.plot_class_distribution(y, save_path='class_dist.png')

# Split data
splits = splitter.split_data(X, y, test_size=0.2, val_size=0.2)

# Verify class balance
distributions = splitter.verify_class_balance(splits)

# Save splits
splitter.save_splits(splits, 'output_dir', prefix='experiment1_')
```

## Configuration

You can customize the splitting process using a YAML configuration file. Example:

```yaml
split_ratios:
  test_size: 0.2
  validation_size: 0.2

random_state: 42
stratify: true

balance_method: none  # Options: none, undersample, oversample

output:
  save_format: csv
  include_index: false
  compression: none

logging:
  level: INFO
  file: split_log.txt
```

## Input Data Format

The tool accepts CSV and Excel files with the following requirements:
- One column must be designated as the target variable
- All other columns will be treated as features
- No missing values in the target column
- Target column should contain class labels

## Output

The tool generates:
1. Three split datasets:
   - train.csv
   - val.csv
   - test.csv
2. Class distribution visualization
3. Detailed log file
4. Class distribution statistics for each split

## License

This project is licensed under the MIT License. 