import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

class EcommerceDataCleaner:
    def __init__(self, input_file):
        """Initialize the cleaner with input file path."""
        self.input_file = input_file
        self.data = None
        self.cleaning_report = []
        
    def load_data(self):
        """Import raw e-commerce data."""
        try:
            self.data = pd.read_csv(self.input_file)
            self.cleaning_report.append(f"Successfully loaded {len(self.data)} records with {len(self.data.columns)} columns")
            return True
        except Exception as e:
            self.cleaning_report.append(f"Error loading data: {str(e)}")
            return False
    
    def handle_missing_entries(self):
        """Detect and handle missing entries."""
        if self.data is None:
            return
        
        # Get initial missing value counts
        missing_before = self.data.isnull().sum()
        self.cleaning_report.append("\nMissing values before cleaning:")
        for col in missing_before.index:
            if missing_before[col] > 0:
                self.cleaning_report.append(f"- {col}: {missing_before[col]} missing values")
        
        # Handle missing values based on data type
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(exclude=[np.number]).columns
        
        # For numeric columns: fill with median
        for col in numeric_cols:
            if self.data[col].isnull().sum() > 0:
                median_val = self.data[col].median()
                self.data[col].fillna(median_val, inplace=True)
                self.cleaning_report.append(f"Filled missing values in {col} with median: {median_val:.2f}")
        
        # For categorical columns: fill with mode
        for col in categorical_cols:
            if self.data[col].isnull().sum() > 0:
                mode_val = self.data[col].mode()[0]
                self.data[col].fillna(mode_val, inplace=True)
                self.cleaning_report.append(f"Filled missing values in {col} with mode: {mode_val}")
        
        # Get final missing value counts
        missing_after = self.data.isnull().sum()
        self.cleaning_report.append("\nMissing values after cleaning:")
        for col in missing_after.index:
            if missing_after[col] > 0:
                self.cleaning_report.append(f"- {col}: {missing_after[col]} missing values")
    
    def handle_outliers(self, columns=None, threshold=3):
        """Identify outliers in price and quantity columns using Z-score method."""
        if self.data is None:
            return
        
        if columns is None:
            # Default to price and quantity columns if they exist
            columns = [col for col in self.data.columns if col.lower() in ['price', 'quantity', 'amount']]
        
        # Create directory for plots
        os.makedirs('outlier_plots', exist_ok=True)
        
        for col in columns:
            if col in self.data.columns and self.data[col].dtype in ['int64', 'float64']:
                # Calculate z-scores
                z_scores = np.abs(stats.zscore(self.data[col]))
                outliers = z_scores > threshold
                outlier_count = sum(outliers)
                
                if outlier_count > 0:
                    self.cleaning_report.append(f"\nFound {outlier_count} outliers in {col}")
                    
                    # Create before/after plots
                    plt.figure(figsize=(12, 5))
                    
                    # Before plot
                    plt.subplot(1, 2, 1)
                    sns.boxplot(y=self.data[col])
                    plt.title(f"Before Outlier Treatment: {col}")
                    
                    # Calculate bounds
                    mean = self.data[col].mean()
                    std = self.data[col].std()
                    lower_bound = mean - threshold * std
                    upper_bound = mean + threshold * std
                    
                    # Cap outliers
                    self.data[col] = self.data[col].clip(lower=lower_bound, upper=upper_bound)
                    
                    # After plot
                    plt.subplot(1, 2, 2)
                    sns.boxplot(y=self.data[col])
                    plt.title(f"After Outlier Treatment: {col}")
                    
                    # Save plot
                    plt.tight_layout()
                    plt.savefig(f'outlier_plots/outliers_{col}.png')
                    plt.close()
                    
                    self.cleaning_report.append(f"- Capped values between {lower_bound:.2f} and {upper_bound:.2f}")
                    self.cleaning_report.append(f"- Plot saved as outliers_{col}.png")
    
    def save_cleaned_data(self, output_file):
        """Save the cleaned data."""
        if self.data is None:
            return
        
        try:
            self.data.to_csv(output_file, index=False)
            self.cleaning_report.append(f"\nCleaned data saved to {output_file}")
            return True
        except Exception as e:
            self.cleaning_report.append(f"\nError saving data: {str(e)}")
            return False
    
    def save_report(self, report_file='cleaning_report.txt'):
        """Save the cleaning report."""
        try:
            with open(report_file, 'w') as f:
                f.write("E-COMMERCE DATA CLEANING REPORT\n")
                f.write("=" * 50 + "\n")
                f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("\n".join(self.cleaning_report))
            return True
        except Exception as e:
            print(f"Error saving report: {str(e)}")
            return False

def main():
    # Example usage
    input_file = "sample_ecommerce_data.csv"
    
    print("E-COMMERCE DATA CLEANER")
    print("=" * 50)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Please place your e-commerce data CSV file in the same directory.")
        return
    
    # Initialize cleaner
    cleaner = EcommerceDataCleaner(input_file)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    if not cleaner.load_data():
        print("Failed to load data. Please check the input file.")
        return
    
    # Step 2: Handle missing entries
    print("2. Handling missing entries...")
    cleaner.handle_missing_entries()
    
    # Step 3: Handle outliers
    print("3. Detecting and handling outliers...")
    cleaner.handle_outliers()
    
    # Step 4: Save cleaned data
    print("4. Saving cleaned data...")
    if cleaner.save_cleaned_data("cleaned_ecommerce_data.csv"):
        print("✓ Cleaned data saved successfully!")
    
    # Save cleaning report
    if cleaner.save_report():
        print("✓ Cleaning report generated successfully!")
    
    print("\nCleaning process completed!")
    print("- Cleaned data saved as: cleaned_ecommerce_data.csv")
    print("- Cleaning report saved as: cleaning_report.txt")
    print("- Outlier plots saved in: outlier_plots/")

if __name__ == "__main__":
    main() 