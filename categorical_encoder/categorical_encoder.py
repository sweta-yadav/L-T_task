import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os

class CategoricalDataEncoder:
    def __init__(self):
        """Initialize the categorical data encoder."""
        self.data = None
        self.categorical_columns = []
        self.label_encoders = {}
        self.onehot_encoder = None
        self.encoding_report = []
        
    def load_data(self, file_path):
        """Load dataset from CSV file."""
        try:
            self.data = pd.read_csv(file_path)
            self.encoding_report.append(f"Successfully loaded {len(self.data)} records with {len(self.data.columns)} columns")
            return True
        except Exception as e:
            self.encoding_report.append(f"Error loading data: {str(e)}")
            return False
    
    def identify_categorical_columns(self):
        """Identify categorical columns in the dataset."""
        if self.data is None:
            return []
        
        # Identify columns with object or category dtype
        self.categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self.encoding_report.append("\nIdentified categorical columns:")
        for col in self.categorical_columns:
            unique_values = self.data[col].nunique()
            self.encoding_report.append(f"- {col}: {unique_values} unique values")
        
        return self.categorical_columns
    
    def apply_label_encoding(self, columns=None):
        """Apply label encoding to specified columns."""
        if self.data is None:
            return
        
        if columns is None:
            columns = self.categorical_columns
        
        self.encoding_report.append("\nApplying label encoding:")
        
        for col in columns:
            if col in self.data.columns:
                try:
                    le = LabelEncoder()
                    self.data[f"{col}_encoded"] = le.fit_transform(self.data[col])
                    self.label_encoders[col] = le
                    
                    # Report mapping
                    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                    self.encoding_report.append(f"\n{col} encoding mapping:")
                    for original, encoded in mapping.items():
                        self.encoding_report.append(f"  {original} → {encoded}")
                except Exception as e:
                    self.encoding_report.append(f"Error encoding {col}: {str(e)}")
    
    def apply_onehot_encoding(self, columns=None):
        """Apply one-hot encoding to specified columns."""
        if self.data is None:
            return
        
        if columns is None:
            columns = self.categorical_columns
        
        self.encoding_report.append("\nApplying one-hot encoding:")
        
        try:
            # Create one-hot encoder
            self.onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            
            # Fit and transform the data
            onehot_data = self.onehot_encoder.fit_transform(self.data[columns])
            
            # Get feature names
            feature_names = []
            for i, col in enumerate(columns):
                categories = self.onehot_encoder.categories_[i]
                for cat in categories:
                    feature_names.append(f"{col}_{cat}")
            
            # Convert to DataFrame
            onehot_df = pd.DataFrame(onehot_data, columns=feature_names)
            
            # Drop original columns and concatenate one-hot encoded columns
            self.data = pd.concat([self.data.drop(columns=columns), onehot_df], axis=1)
            
            self.encoding_report.append(f"Successfully created {len(feature_names)} one-hot encoded features")
            
        except Exception as e:
            self.encoding_report.append(f"Error during one-hot encoding: {str(e)}")
    
    def save_encoded_data(self, output_file):
        """Save the encoded data to CSV."""
        if self.data is None:
            return False
        
        try:
            self.data.to_csv(output_file, index=False)
            self.encoding_report.append(f"\nEncoded data saved to {output_file}")
            return True
        except Exception as e:
            self.encoding_report.append(f"\nError saving data: {str(e)}")
            return False
    
    def save_report(self, report_file='encoding_report.txt'):
        """Save the encoding report."""
        try:
            with open(report_file, 'w') as f:
                f.write("CATEGORICAL DATA ENCODING REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write("\n".join(self.encoding_report))
            return True
        except Exception as e:
            print(f"Error saving report: {str(e)}")
            return False

def main():
    print("CATEGORICAL DATA ENCODER")
    print("=" * 50)
    
    # Initialize encoder
    encoder = CategoricalDataEncoder()
    
    # Get input file
    input_file = input("\nEnter the path to your CSV file: ")
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return
    
    # Load data
    print("\n1. Loading data...")
    if not encoder.load_data(input_file):
        print("Failed to load data. Please check the input file.")
        return
    
    # Identify categorical columns
    print("2. Identifying categorical columns...")
    categorical_cols = encoder.identify_categorical_columns()
    if not categorical_cols:
        print("No categorical columns found in the dataset.")
        return
    
    # Choose encoding method
    print("\n3. Choose encoding method:")
    print("1. Label Encoding")
    print("2. One-Hot Encoding")
    choice = input("Enter your choice (1 or 2): ")
    
    # Apply chosen encoding
    if choice == "1":
        print("\nApplying label encoding...")
        encoder.apply_label_encoding()
    elif choice == "2":
        print("\nApplying one-hot encoding...")
        encoder.apply_onehot_encoding()
    else:
        print("Invalid choice. Please enter 1 or 2.")
        return
    
    # Save encoded data
    output_file = input_file.replace(".csv", "_encoded.csv")
    print("\n4. Saving encoded data...")
    if encoder.save_encoded_data(output_file):
        print("✓ Encoded data saved successfully!")
    
    # Save encoding report
    report_file = "encoding_report.txt"
    if encoder.save_report(report_file):
        print("✓ Encoding report generated successfully!")
    
    print("\nEncoding process completed!")
    print(f"- Encoded data saved as: {output_file}")
    print(f"- Encoding report saved as: {report_file}")

if __name__ == "__main__":
    main() 