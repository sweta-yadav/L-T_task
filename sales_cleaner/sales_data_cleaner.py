import pandas as pd
import numpy as np
from scipy import stats

class SalesDataCleaner:
    def __init__(self, input_file):
        """Initialize the cleaner with input file path."""
        self.input_file = input_file
        self.data = None
    
    def load_data(self):
        """Load sales data using Pandas."""
        try:
            self.data = pd.read_csv(self.input_file)
            print(f"Successfully loaded data with {len(self.data)} rows.")
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def handle_missing_values(self, strategy='mean'):
        """Identify and handle missing values using specified strategy."""
        if self.data is None:
            print("Please load data first.")
            return
        
        # Get missing value statistics
        missing_stats = self.data.isnull().sum()
        print("\nMissing values per column:")
        print(missing_stats[missing_stats > 0])
        
        # Handle missing values based on data type
        for column in self.data.columns:
            if self.data[column].dtype in ['int64', 'float64']:
                if strategy == 'mean':
                    self.data[column].fillna(self.data[column].mean(), inplace=True)
                elif strategy == 'median':
                    self.data[column].fillna(self.data[column].median(), inplace=True)
            else:
                # For categorical/text columns, fill with mode
                self.data[column].fillna(self.data[column].mode()[0], inplace=True)
        
        print("Missing values have been handled.")
    
    def remove_duplicates(self):
        """Detect and remove duplicate records."""
        if self.data is None:
            print("Please load data first.")
            return
        
        initial_rows = len(self.data)
        self.data.drop_duplicates(inplace=True)
        removed_rows = initial_rows - len(self.data)
        print(f"\nRemoved {removed_rows} duplicate rows.")
    
    def detect_outliers(self, columns, z_threshold=3):
        """Apply outlier detection using Z-score method."""
        if self.data is None:
            print("Please load data first.")
            return
        
        outliers_removed = 0
        for column in columns:
            if self.data[column].dtype in ['int64', 'float64']:
                z_scores = np.abs(stats.zscore(self.data[column]))
                outliers_mask = z_scores < z_threshold
                outliers_count = len(self.data) - sum(outliers_mask)
                print(f"\nFound {outliers_count} outliers in column {column}")
                
                # Remove outliers
                self.data = self.data[outliers_mask]
                outliers_removed += outliers_count
        
        print(f"\nTotal outliers removed: {outliers_removed}")
    
    def save_cleaned_data(self, output_file):
        """Save the cleaned dataset as a new CSV file."""
        if self.data is None:
            print("Please load data first.")
            return
        
        try:
            self.data.to_csv(output_file, index=False)
            print(f"\nCleaned data saved successfully to {output_file}")
            print(f"Final dataset shape: {self.data.shape}")
        except Exception as e:
            print(f"Error saving data: {str(e)}")

def main():
    # Example usage
    input_file = "sales_data.csv"  # Replace with your input file
    output_file = "cleaned_sales_data.csv"
    
    # Initialize the cleaner
    cleaner = SalesDataCleaner(input_file)
    
    # Execute cleaning pipeline
    if cleaner.load_data():
        cleaner.handle_missing_values(strategy='mean')
        cleaner.remove_duplicates()
        
        # Specify numerical columns for outlier detection
        numerical_columns = [col for col in cleaner.data.columns 
                           if cleaner.data[col].dtype in ['int64', 'float64']]
        cleaner.detect_outliers(numerical_columns)
        
        cleaner.save_cleaned_data(output_file)

if __name__ == "__main__":
    main() 