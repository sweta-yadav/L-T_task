import pandas as pd
import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def main():
    print("=" * 80)
    print("SALES DATA CLEANING AND PREPROCESSING TOOL")
    print("=" * 80)
    
    # Step 1: Load the data
    print("\nStep 1: Loading data...")
    try:
        # Check if the data file exists
        file_path = 'superstore_sales_data.csv'
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found.")
            print("Please make sure the CSV file is in the same directory as this script.")
            return
        
        sales_data = pd.read_csv(file_path)
        print(f"Data loaded successfully with {sales_data.shape[0]} rows and {sales_data.shape[1]} columns.")
        
        # Display sample data
        print("\nFirst 5 rows of the dataset:")
        print(sales_data.head())
        
        # Display data info
        print("\nDataset information:")
        print(sales_data.info())
        
        # Display summary statistics
        print("\nSummary statistics for numerical columns:")
        print(sales_data.describe())
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Step 2: Handle missing values
    print("\nStep 2: Handling missing values...")
    print("Missing values in each column before cleaning:")
    print(sales_data.isnull().sum())
    
    # For numeric columns: fill with median
    numeric_cols = sales_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if sales_data[col].isnull().sum() > 0:
            median_value = sales_data[col].median()
            sales_data[col] = sales_data[col].fillna(median_value)
            print(f"  - Filled {col} missing values with median: {median_value}")
    
    # For categorical columns: fill with mode
    categorical_cols = sales_data.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        if sales_data[col].isnull().sum() > 0:
            mode_value = sales_data[col].mode()[0]
            sales_data[col] = sales_data[col].fillna(mode_value)
            print(f"  - Filled {col} missing values with mode: {mode_value}")
    
    print("\nMissing values in each column after cleaning:")
    print(sales_data.isnull().sum())
    
    # Step 3: Convert date columns to datetime
    print("\nStep 3: Converting date columns to datetime format...")
    date_columns = ['Order Date', 'Ship Date']
    for col in date_columns:
        if col in sales_data.columns:
            try:
                sales_data[col] = pd.to_datetime(sales_data[col])
                print(f"  - Converted {col} to datetime format")
            except:
                print(f"  - Failed to convert {col} to datetime format")
    
    # Step 4: Standardize categorical columns
    print("\nStep 4: Standardizing categorical columns...")
    cat_columns = ['Ship Mode', 'Segment', 'Category', 'Sub-Category']
    for col in cat_columns:
        if col in sales_data.columns:
            # Convert to proper case (first letter capitalized)
            if sales_data[col].dtype == 'object':
                # Check for inconsistent formatting
                unique_before = sales_data[col].nunique()
                
                # Standardize format (proper case)
                sales_data[col] = sales_data[col].str.title()
                
                unique_after = sales_data[col].nunique()
                if unique_before != unique_after:
                    print(f"  - Standardized {col}: fixed {unique_before - unique_after} inconsistent values")
    
    # Step 5: Remove duplicates
    print("\nStep 5: Removing duplicate records...")
    duplicates_count = sales_data.duplicated().sum()
    if duplicates_count > 0:
        sales_data = sales_data.drop_duplicates()
        print(f"  - Removed {duplicates_count} duplicate rows")
    else:
        print("  - No duplicate rows found")
    
    # Step 6: Detect and handle outliers in numeric columns
    print("\nStep 6: Detecting and handling outliers...")
    outlier_cols = ['Quantity', 'Sales', 'Discount', 'Profit']
    
    # Create a directory for plots if it doesn't exist
    os.makedirs('data_cleaning_plots', exist_ok=True)
    
    for col in outlier_cols:
        if col in sales_data.columns:
            # Detect outliers using Z-score
            z_scores = np.abs(stats.zscore(sales_data[col]))
            outliers = z_scores > 3
            outlier_count = np.sum(outliers)
            
            if outlier_count > 0:
                print(f"  - Found {outlier_count} outliers in '{col}' column")
                
                # Plot before handling outliers
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                sns.boxplot(y=sales_data[col])
                plt.title(f"Before outlier handling: {col}")
                
                # Cap outliers
                upper_limit = sales_data[col].mean() + 3 * sales_data[col].std()
                lower_limit = sales_data[col].mean() - 3 * sales_data[col].std()
                
                # Store original values
                original_values = sales_data.loc[outliers, col].copy()
                
                # Apply capping
                sales_data[col] = sales_data[col].clip(lower=lower_limit, upper=upper_limit)
                
                # Plot after handling outliers
                plt.subplot(1, 2, 2)
                sns.boxplot(y=sales_data[col])
                plt.title(f"After outlier handling: {col}")
                
                # Save the plot
                plt.tight_layout()
                plt.savefig(f'data_cleaning_plots/outlier_handling_{col}.png')
                plt.close()
                
                print(f"    - Capped outliers between {lower_limit:.2f} and {upper_limit:.2f}")
                print(f"    - Plot saved to data_cleaning_plots/outlier_handling_{col}.png")
            else:
                print(f"  - No outliers found in '{col}' column")
    
    # Step 7: Save the cleaned dataset
    print("\nStep 7: Saving the cleaned dataset...")
    output_file = 'cleaned_superstore_sales_data.csv'
    sales_data.to_csv(output_file, index=False)
    print(f"  - Cleaned data saved as '{output_file}'")
    
    # Step 8: Generate a data cleaning report
    print("\nStep 8: Generating data cleaning report...")
    report_file = 'data_cleaning_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SALES DATA CLEANING REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("1. DATA SUMMARY\n")
        f.write(f"   - Original rows: {sales_data.shape[0] + duplicates_count}\n")
        f.write(f"   - Cleaned rows: {sales_data.shape[0]}\n")
        f.write(f"   - Columns: {sales_data.shape[1]}\n\n")
        
        f.write("2. CLEANING ACTIONS\n")
        f.write(f"   - Removed {duplicates_count} duplicate records\n")
        
        f.write("\n3. COLUMN STATISTICS AFTER CLEANING\n")
        for col in sales_data.columns:
            if sales_data[col].dtype in ['int64', 'float64']:
                f.write(f"   - {col}: Min={sales_data[col].min():.2f}, Max={sales_data[col].max():.2f}, ")
                f.write(f"Mean={sales_data[col].mean():.2f}, Median={sales_data[col].median():.2f}\n")
            else:
                f.write(f"   - {col}: {sales_data[col].nunique()} unique values\n")
    
    print(f"  - Cleaning report saved as '{report_file}'")
    
    print("\nData cleaning process completed successfully!")
    print(f"You can find the cleaned data in '{output_file}'")
    print(f"A detailed cleaning report is available in '{report_file}'")
    print("Boxplots showing outlier handling are saved in the 'data_cleaning_plots' directory")

if __name__ == "__main__":
    main() 