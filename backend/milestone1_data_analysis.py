"""
Milestone 1: Categorical and Numerical Data Analysis
This script performs comprehensive analysis of the visa dataset,
separating categorical and numerical features for detailed insights.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

class DataAnalyzer:
    def __init__(self, data_path):
        """Initialize the data analyzer with dataset path"""
        self.data_path = data_path
        self.df = None
        self.categorical_features = []
        self.numerical_features = []
        
    def load_data(self):
        """Load the dataset"""
        print("=" * 80)
        print("LOADING DATASET")
        print("=" * 80)
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print()
        
    def identify_feature_types(self):
        """Automatically identify categorical and numerical features"""
        print("=" * 80)
        print("IDENTIFYING FEATURE TYPES")
        print("=" * 80)
        
        for col in self.df.columns:
            if self.df[col].dtype in ['object', 'category']:
                self.categorical_features.append(col)
            elif self.df[col].dtype in ['int64', 'float64']:
                self.numerical_features.append(col)
        
        print(f"\nCategorical Features ({len(self.categorical_features)}):")
        for feat in self.categorical_features:
            print(f"  - {feat}")
        
        print(f"\nNumerical Features ({len(self.numerical_features)}):")
        for feat in self.numerical_features:
            print(f"  - {feat}")
        print()
        
    def analyze_numerical_features(self):
        """Perform comprehensive numerical data analysis"""
        print("=" * 80)
        print("NUMERICAL FEATURES ANALYSIS")
        print("=" * 80)
        
        if not self.numerical_features:
            print("No numerical features found.")
            return {}
        
        numerical_analysis = {}
        
        # Basic statistics
        print("\n1. DESCRIPTIVE STATISTICS")
        print("-" * 80)
        stats = self.df[self.numerical_features].describe()
        print(stats.to_string())
        
        # Individual feature analysis
        print("\n2. DETAILED FEATURE ANALYSIS")
        print("-" * 80)
        
        for feature in self.numerical_features:
            print(f"\n{feature.upper()}:")
            data = self.df[feature]
            
            analysis = {
                'count': len(data),
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'range': data.max() - data.min(),
                'q1': data.quantile(0.25),
                'q3': data.quantile(0.75),
                'iqr': data.quantile(0.75) - data.quantile(0.25),
                'missing_values': data.isnull().sum(),
                'missing_percentage': (data.isnull().sum() / len(data)) * 100
            }
            
            print(f"  Count: {analysis['count']}")
            print(f"  Mean: {analysis['mean']:.2f}")
            print(f"  Median: {analysis['median']:.2f}")
            print(f"  Std Dev: {analysis['std']:.2f}")
            print(f"  Min: {analysis['min']:.2f}")
            print(f"  Max: {analysis['max']:.2f}")
            print(f"  Range: {analysis['range']:.2f}")
            print(f"  Q1 (25%): {analysis['q1']:.2f}")
            print(f"  Q3 (75%): {analysis['q3']:.2f}")
            print(f"  IQR: {analysis['iqr']:.2f}")
            print(f"  Missing Values: {analysis['missing_values']} ({analysis['missing_percentage']:.2f}%)")
            
            # Skewness and Kurtosis
            skewness = data.skew()
            kurtosis = data.kurt()
            print(f"  Skewness: {skewness:.4f}")
            print(f"  Kurtosis: {kurtosis:.4f}")
            
            analysis['skewness'] = skewness
            analysis['kurtosis'] = kurtosis
            
            numerical_analysis[feature] = analysis
        
        # Identify outliers
        print("\n3. OUTLIER DETECTION (IQR METHOD)")
        print("-" * 80)
        
        for feature in self.numerical_features:
            q1 = self.df[feature].quantile(0.25)
            q3 = self.df[feature].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = self.df[(self.df[feature] < lower_bound) | (self.df[feature] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(self.df)) * 100
            
            print(f"{feature}:")
            print(f"  Lower Bound: {lower_bound:.2f}")
            print(f"  Upper Bound: {upper_bound:.2f}")
            print(f"  Outliers: {outlier_count} ({outlier_percentage:.2f}%)")
        
        print()
        return numerical_analysis
    
    def analyze_categorical_features(self):
        """Perform comprehensive categorical data analysis"""
        print("=" * 80)
        print("CATEGORICAL FEATURES ANALYSIS")
        print("=" * 80)
        
        if not self.categorical_features:
            print("No categorical features found.")
            return {}
        
        categorical_analysis = {}
        
        print("\n1. FEATURE SUMMARY")
        print("-" * 80)
        
        for feature in self.categorical_features:
            print(f"\n{feature.upper()}:")
            data = self.df[feature]
            
            # Unique values and counts
            unique_count = data.nunique()
            total_count = len(data)
            missing_count = data.isnull().sum()
            missing_percentage = (missing_count / total_count) * 100
            
            print(f"  Total Records: {total_count}")
            print(f"  Unique Values: {unique_count}")
            print(f"  Missing Values: {missing_count} ({missing_percentage:.2f}%)")
            
            # Value counts
            value_counts = data.value_counts()
            print(f"\n  Value Distribution:")
            for value, count in value_counts.items():
                percentage = (count / total_count) * 100
                print(f"    {value}: {count} ({percentage:.2f}%)")
            
            # Mode
            mode = data.mode()[0] if len(data.mode()) > 0 else None
            print(f"\n  Most Frequent (Mode): {mode}")
            
            analysis = {
                'unique_count': unique_count,
                'total_count': total_count,
                'missing_count': missing_count,
                'missing_percentage': missing_percentage,
                'value_counts': value_counts.to_dict(),
                'mode': mode
            }
            
            categorical_analysis[feature] = analysis
        
        # Cross-tabulation with target variable if exists
        if 'visa_status' in self.categorical_features:
            print("\n2. CROSS-TABULATION WITH VISA STATUS")
            print("-" * 80)
            
            for feature in self.categorical_features:
                if feature != 'visa_status':
                    print(f"\n{feature.upper()} vs VISA_STATUS:")
                    crosstab = pd.crosstab(self.df[feature], self.df['visa_status'], margins=True)
                    print(crosstab.to_string())
        
        print()
        return categorical_analysis
    
    def generate_summary_report(self):
        """Generate overall dataset summary"""
        print("=" * 80)
        print("DATASET SUMMARY REPORT")
        print("=" * 80)
        
        print(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nDataset: {self.data_path}")
        print(f"Total Records: {len(self.df)}")
        print(f"Total Features: {len(self.df.columns)}")
        print(f"  - Categorical: {len(self.categorical_features)}")
        print(f"  - Numerical: {len(self.numerical_features)}")
        
        # Missing data summary
        print("\nMissing Data Summary:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("  No missing values found")
        else:
            for col, count in missing[missing > 0].items():
                percentage = (count / len(self.df)) * 100
                print(f"  {col}: {count} ({percentage:.2f}%)")
        
        # Memory usage
        memory_usage = self.df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"\nMemory Usage: {memory_usage:.2f} MB")
        
        print("\n" + "=" * 80)
        print("MILESTONE 1 ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print()
    
    def run_complete_analysis(self):
        """Execute complete Milestone 1 analysis"""
        self.load_data()
        self.identify_feature_types()
        self.analyze_numerical_features()
        self.analyze_categorical_features()
        self.generate_summary_report()
        
        return {
            'dataframe': self.df,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features
        }


def main():
    """Main execution function"""
    # Path to dataset
    data_path = '../data/visa_data.csv'
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return
    
    # Initialize and run analyzer
    analyzer = DataAnalyzer(data_path)
    results = analyzer.run_complete_analysis()
    
    print("Analysis results have been saved to console output.")
    print("You can redirect this output to a file using: python milestone1_data_analysis.py > analysis_report.txt")


if __name__ == "__main__":
    main()
