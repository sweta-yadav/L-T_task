from data_quality_validator import DataQualityValidator
import os

def main():
    # Create validation_plots directory if it doesn't exist
    if not os.path.exists('validation_plots'):
        os.makedirs('validation_plots')

    # Initialize the validator with the sample dataset
    validator = DataQualityValidator('data_validator/sample_dataset.csv')

    # Run the validation process
    print("Starting data quality validation...")
    
    # Load and summarize the dataset
    validator.load_dataset()
    
    # Run all validation checks
    validator.validate_data_types()
    validator.analyze_missing_values()
    validator.detect_anomalies()
    validator.check_duplicates()
    validator.analyze_value_ranges()
    
    # Generate recommendations and save reports
    validator.generate_recommendations()
    validator.save_report()
    validator.generate_html_report()
    
    print("\nValidation complete! Check data_quality_report.json and data_quality_report.html for results.")

if __name__ == "__main__":
    main() 