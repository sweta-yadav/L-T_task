import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import argparse
import yaml
import logging
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns

class DataSplitter:
    """A class to automate the splitting of datasets into training, testing, and validation subsets."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the DataSplitter with optional configuration.
        
        Args:
            config_path (str, optional): Path to YAML configuration file
        """
        self.logger = self._setup_logger()
        self.config = self._load_config(config_path) if config_path else {}
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path (str): Path to YAML configuration file
            
        Returns:
            Dict: Configuration dictionary
        """
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.warning(f"Error loading config file: {e}")
            return {}
    
    def load_data(self, file_path: str, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load dataset from file and separate features and target.
        
        Args:
            file_path (str): Path to the dataset file
            target_column (str): Name of the target column
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target variables
        """
        try:
            # Determine file type and read accordingly
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")
            
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            self.logger.info(f"Data loaded successfully: {len(df)} samples")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def analyze_class_distribution(self, y: pd.Series) -> Dict:
        """
        Analyze the distribution of classes in the target variable.
        
        Args:
            y (pd.Series): Target variable
            
        Returns:
            Dict: Class distribution statistics
        """
        class_counts = y.value_counts()
        class_proportions = y.value_counts(normalize=True)
        
        stats = {
            'class_counts': class_counts.to_dict(),
            'class_proportions': class_proportions.to_dict(),
            'n_classes': len(class_counts),
            'imbalance_ratio': class_counts.max() / class_counts.min()
        }
        
        self.logger.info(f"Class distribution analysis completed: {stats['n_classes']} classes found")
        return stats
    
    def plot_class_distribution(self, y: pd.Series, save_path: str = None):
        """
        Plot the class distribution.
        
        Args:
            y (pd.Series): Target variable
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x=y)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            self.logger.info(f"Class distribution plot saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def split_data(self, 
                  X: pd.DataFrame, 
                  y: pd.Series,
                  test_size: float = 0.2,
                  val_size: float = 0.2,
                  stratify: bool = True,
                  random_state: int = 42) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """
        Split data into training, testing, and validation sets.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            test_size (float): Proportion of data for testing
            val_size (float): Proportion of training data for validation
            stratify (bool): Whether to maintain class distribution in splits
            random_state (int): Random seed for reproducibility
            
        Returns:
            Dict: Dictionary containing the split datasets
        """
        try:
            stratify_param = y if stratify else None
            
            # First split: training + validation vs testing
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y,
                test_size=test_size,
                stratify=stratify_param,
                random_state=random_state
            )
            
            # Second split: training vs validation
            stratify_param = y_train_val if stratify else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=val_size,
                stratify=stratify_param,
                random_state=random_state
            )
            
            splits = {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test
            }
            
            self.logger.info(f"""
            Data split completed:
            Training set: {len(X_train)} samples
            Validation set: {len(X_val)} samples
            Test set: {len(X_test)} samples
            """)
            
            return splits
            
        except Exception as e:
            self.logger.error(f"Error splitting data: {e}")
            raise
    
    def verify_class_balance(self, splits: Dict[str, Union[pd.DataFrame, pd.Series]]) -> Dict:
        """
        Verify the class balance in all splits.
        
        Args:
            splits (Dict): Dictionary containing the split datasets
            
        Returns:
            Dict: Class distribution statistics for each split
        """
        distributions = {}
        for split_name in ['train', 'val', 'test']:
            y_split = splits[f'y_{split_name}']
            distributions[split_name] = self.analyze_class_distribution(y_split)
        
        return distributions
    
    def save_splits(self, 
                   splits: Dict[str, Union[pd.DataFrame, pd.Series]], 
                   output_dir: str,
                   prefix: str = ''):
        """
        Save the split datasets to files.
        
        Args:
            splits (Dict): Dictionary containing the split datasets
            output_dir (str): Directory to save the files
            prefix (str, optional): Prefix for the output files
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            for split_name in ['train', 'val', 'test']:
                # Combine features and target for each split
                X_split = splits[f'X_{split_name}']
                y_split = splits[f'y_{split_name}']
                
                combined = pd.concat([X_split, y_split], axis=1)
                
                # Save to CSV
                output_path = os.path.join(output_dir, f"{prefix}{split_name}.csv")
                combined.to_csv(output_path, index=False)
                
                self.logger.info(f"Saved {split_name} set to {output_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving splits: {e}")
            raise

def main():
    """Main function to run the data splitter from command line."""
    parser = argparse.ArgumentParser(description='Data Split Automation Tool')
    parser.add_argument('--input', required=True, help='Path to input dataset')
    parser.add_argument('--target', required=True, help='Name of target column')
    parser.add_argument('--output-dir', required=True, help='Output directory for split datasets')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of data for testing')
    parser.add_argument('--val-size', type=float, default=0.2, help='Proportion of training data for validation')
    parser.add_argument('--no-stratify', action='store_true', help='Disable stratified splitting')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    parser.add_argument('--prefix', default='', help='Prefix for output files')
    
    args = parser.parse_args()
    
    # Initialize DataSplitter
    splitter = DataSplitter(args.config)
    
    # Load data
    X, y = splitter.load_data(args.input, args.target)
    
    # Analyze and plot initial class distribution
    splitter.analyze_class_distribution(y)
    splitter.plot_class_distribution(y, os.path.join(args.output_dir, f"{args.prefix}class_distribution.png"))
    
    # Split data
    splits = splitter.split_data(
        X, y,
        test_size=args.test_size,
        val_size=args.val_size,
        stratify=not args.no_stratify,
        random_state=args.random_state
    )
    
    # Verify class balance
    distributions = splitter.verify_class_balance(splits)
    
    # Save splits
    splitter.save_splits(splits, args.output_dir, args.prefix)

if __name__ == '__main__':
    main() 