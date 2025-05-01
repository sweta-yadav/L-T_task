import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
from typing import Dict, List, Any

class DataQualityValidator:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.validation_results = {
            "data_types": {},
            "missing_values": {},
            "anomalies": {},
            "duplicates": {},
            "value_ranges": {},
            "recommendations": []
        }
        
        # Define expected data types and value ranges
        self.expected_types = {
            'customer_id': 'int64',
            'age': 'int64',
            'income': 'float64',
            'education': 'object',
            'occupation': 'object',
            'loan_amount': 'float64',
            'credit_score': 'int64',
            'purchase_frequency': 'object'
        }
        
        self.valid_ranges = {
            'age': (18, 100),
            'income': (0, 1000000),
            'loan_amount': (0, 1000000),
            'credit_score': (300, 850)
        }
        
        self.valid_categories = {
            'education': ['High School', 'Bachelor', 'Master', 'PhD'],
            'purchase_frequency': ['Low', 'Medium', 'High']
        }

    def load_dataset(self):
        """Load and perform initial dataset analysis."""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Dataset loaded successfully with {len(self.df)} rows and {len(self.df.columns)} columns")
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")

    def validate_data_types(self):
        """Check if columns have the expected data types."""
        for column, expected_type in self.expected_types.items():
            if column not in self.df.columns:
                self.validation_results["data_types"][column] = "Column missing"
                continue
                
            current_type = str(self.df[column].dtype)
            is_valid = current_type == expected_type
            
            self.validation_results["data_types"][column] = {
                "expected": expected_type,
                "current": current_type,
                "is_valid": is_valid
            }
            
            # Create visualization for numeric columns
            if expected_type in ['int64', 'float64']:
                self._create_distribution_plot(column)

    def analyze_missing_values(self):
        """Analyze missing values in the dataset."""
        missing_counts = self.df.isnull().sum()
        missing_percentages = (missing_counts / len(self.df)) * 100
        
        for column in self.df.columns:
            self.validation_results["missing_values"][column] = {
                "count": int(missing_counts[column]),
                "percentage": float(missing_percentages[column])
            }
        
        # Create missing values heatmap
        self._create_missing_values_heatmap()

    def detect_anomalies(self):
        """Detect anomalies in numeric columns using IQR method."""
        numeric_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        for column in numeric_columns:
            if column == 'customer_id':  # Skip ID column
                continue
                
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            anomalies = self.df[(self.df[column] < lower_bound) | 
                               (self.df[column] > upper_bound)][column]
            
            self.validation_results["anomalies"][column] = {
                "count": len(anomalies),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "anomalous_values": anomalies.tolist()
            }

    def check_duplicates(self):
        """Check for duplicate records in the dataset."""
        duplicates = self.df.duplicated()
        duplicate_rows = self.df[duplicates]
        
        self.validation_results["duplicates"] = {
            "total_count": int(duplicates.sum()),
            "duplicate_indices": duplicate_rows.index.tolist()
        }

    def analyze_value_ranges(self):
        """Check if values are within expected ranges."""
        for column, (min_val, max_val) in self.valid_ranges.items():
            if column not in self.df.columns:
                continue
                
            # Convert to numeric, coerce invalid values to NaN
            numeric_values = pd.to_numeric(self.df[column], errors='coerce')
            invalid_mask = numeric_values.isna() & ~self.df[column].isna()  # Identify invalid non-missing values
            out_of_range_mask = (numeric_values < min_val) | (numeric_values > max_val)
            
            invalid_values = self.df[column][invalid_mask].tolist()
            out_of_range_values = numeric_values[out_of_range_mask].tolist()
            
            self.validation_results["value_ranges"][column] = {
                "min_expected": min_val,
                "max_expected": max_val,
                "invalid_format_count": len(invalid_values),
                "invalid_format_values": invalid_values,
                "out_of_range_count": len(out_of_range_values),
                "out_of_range_values": out_of_range_values
            }
        
        # Check categorical variables
        for column, valid_categories in self.valid_categories.items():
            if column not in self.df.columns:
                continue
                
            invalid_categories = self.df[~self.df[column].isin(valid_categories)][column].unique()
            
            self.validation_results["value_ranges"][column] = {
                "valid_categories": valid_categories,
                "invalid_categories": invalid_categories.tolist(),
                "invalid_count": len(invalid_categories)
            }

    def generate_recommendations(self):
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Check data types
        for column, result in self.validation_results["data_types"].items():
            if isinstance(result, dict) and not result["is_valid"]:
                recommendations.append(
                    f"Convert {column} from {result['current']} to {result['expected']}"
                )
        
        # Check missing values
        for column, result in self.validation_results["missing_values"].items():
            if result["percentage"] > 5:
                recommendations.append(
                    f"Address missing values in {column} ({result['percentage']:.1f}% missing)"
                )
        
        # Check anomalies
        for column, result in self.validation_results["anomalies"].items():
            if result["count"] > 0:
                recommendations.append(
                    f"Investigate {result['count']} anomalies in {column}"
                )
        
        # Check duplicates
        if self.validation_results["duplicates"]["total_count"] > 0:
            recommendations.append(
                f"Remove {self.validation_results['duplicates']['total_count']} duplicate records"
            )
        
        # Check value ranges
        for column, result in self.validation_results["value_ranges"].items():
            if "invalid_count" in result and result["invalid_count"] > 0:
                recommendations.append(
                    f"Fix {result['invalid_count']} invalid values in {column}"
                )
        
        self.validation_results["recommendations"] = recommendations

    def save_report(self):
        """Save validation results to a JSON file."""
        with open('data_quality_report.json', 'w') as f:
            json.dump(self.validation_results, f, indent=4)

    def generate_html_report(self):
        """Generate an HTML report with validation results."""
        html_content = """
        <html>
        <head>
            <title>Data Quality Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2c3e50; }
                h2 { color: #34495e; }
                .section { margin: 20px 0; padding: 10px; border: 1px solid #bdc3c7; }
                .recommendation { color: #e74c3c; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
        """
        
        # Add title
        html_content += "<h1>Data Quality Validation Report</h1>"
        
        # Add summary section
        html_content += f"""
        <div class='section'>
            <h2>Dataset Summary</h2>
            <p>Total rows: {len(self.df)}</p>
            <p>Total columns: {len(self.df.columns)}</p>
        </div>
        """
        
        # Add recommendations section
        html_content += """
        <div class='section'>
            <h2>Recommendations</h2>
            <ul>
        """
        for rec in self.validation_results["recommendations"]:
            html_content += f"<li class='recommendation'>{rec}</li>"
        html_content += "</ul></div>"
        
        # Add detailed results
        sections = ["data_types", "missing_values", "anomalies", "duplicates", "value_ranges"]
        for section in sections:
            html_content += f"""
            <div class='section'>
                <h2>{section.replace('_', ' ').title()}</h2>
                <table>
                    <tr>
                        <th>Column/Metric</th>
                        <th>Details</th>
                    </tr>
            """
            
            for key, value in self.validation_results[section].items():
                html_content += f"""
                    <tr>
                        <td>{key}</td>
                        <td>{str(value)}</td>
                    </tr>
                """
            
            html_content += "</table></div>"
        
        html_content += "</body></html>"
        
        with open('data_quality_report.html', 'w') as f:
            f.write(html_content)

    def _create_distribution_plot(self, column: str):
        """Create distribution plot for numeric columns."""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x=column, kde=True)
        plt.title(f'Distribution of {column}')
        plt.savefig(f'validation_plots/{column}_distribution.png')
        plt.close()

    def _create_missing_values_heatmap(self):
        """Create heatmap of missing values."""
        plt.figure(figsize=(12, 6))
        sns.heatmap(self.df.isnull(), cmap='viridis', yticklabels=False)
        plt.title('Missing Values Heatmap')
        plt.savefig('validation_plots/missing_values_heatmap.png')
        plt.close()

def main():
    print("=" * 50)
    print("DATA QUALITY VALIDATOR")
    print("=" * 50)
    
    # Example usage
    input_file = "sample_dataset.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Please place your dataset CSV file in the same directory.")
        return
    
    # Initialize validator
    validator = DataQualityValidator(input_file)
    
    # Step 1: Load dataset
    print("\n1. Loading dataset...")
    if not validator.load_dataset():
        print("Failed to load dataset. Please check the input file.")
        return
    
    # Step 2: Validate data types
    print("2. Validating data types...")
    validator.validate_data_types()
    
    # Step 3: Analyze missing values
    print("3. Analyzing missing values...")
    validator.analyze_missing_values()
    
    # Step 4: Detect anomalies
    print("4. Detecting anomalies...")
    validator.detect_anomalies()
    
    # Step 5: Check duplicates
    print("5. Checking for duplicates...")
    validator.check_duplicates()
    
    # Step 6: Analyze value ranges
    print("6. Analyzing value ranges...")
    validator.analyze_value_ranges()
    
    # Step 7: Generate recommendations
    print("7. Generating recommendations...")
    validator.generate_recommendations()
    
    # Save reports
    print("\n8. Saving reports...")
    if validator.save_report():
        print("✓ JSON report saved as: data_quality_report.json")
    
    if validator.generate_html_report():
        print("✓ HTML report saved as: data_quality_report.html")
    
    print("\nValidation completed!")
    print("- Detailed JSON report: data_quality_report.json")
    print("- HTML report: data_quality_report.html")
    print("- Visualization plots: validation_plots/")

if __name__ == "__main__":
    main() 