"""
Milestone 2: Exploratory Data Analysis (EDA) and Feature Engineering
This script performs comprehensive EDA with visualizations and feature engineering
for the visa prediction model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class EDAFeatureEngineering:
    def __init__(self, data_path):
        """Initialize EDA and Feature Engineering"""
        self.data_path = data_path
        self.df = None
        self.output_dir = '../visualizations'
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created visualizations directory: {self.output_dir}")
        
        # Set style for visualizations
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def load_data(self):
        """Load the dataset"""
        print("=" * 80)
        print("LOADING DATASET FOR EDA")
        print("=" * 80)
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print()
        
    def univariate_analysis(self):
        """Perform univariate analysis with visualizations"""
        print("=" * 80)
        print("1. UNIVARIATE ANALYSIS")
        print("=" * 80)
        
        # Identify feature types
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Numerical features - Distribution plots
        print("\nGenerating distribution plots for numerical features...")
        if numerical_cols:
            n_cols = 3
            n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes]
            
            for idx, col in enumerate(numerical_cols):
                ax = axes[idx]
                self.df[col].hist(bins=30, edgecolor='black', ax=ax)
                ax.set_title(f'Distribution of {col}', fontsize=10, fontweight='bold')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for idx in range(len(numerical_cols), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/numerical_distributions.png', dpi=300, bbox_inches='tight')
            print(f"  Saved: numerical_distributions.png")
            plt.close()
        
        # Categorical features - Bar plots
        print("\nGenerating bar plots for categorical features...")
        if categorical_cols:
            n_cols = 2
            n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes]
            
            for idx, col in enumerate(categorical_cols):
                ax = axes[idx]
                value_counts = self.df[col].value_counts()
                value_counts.plot(kind='bar', ax=ax, edgecolor='black')
                ax.set_title(f'Distribution of {col}', fontsize=10, fontweight='bold')
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for idx in range(len(categorical_cols), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/categorical_distributions.png', dpi=300, bbox_inches='tight')
            print(f"  Saved: categorical_distributions.png")
            plt.close()
        
        print("\nUnivariate analysis completed.\n")
    
    def bivariate_analysis(self):
        """Perform bivariate analysis"""
        print("=" * 80)
        print("2. BIVARIATE ANALYSIS")
        print("=" * 80)
        
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Correlation matrix
        if len(numerical_cols) > 1:
            print("\nGenerating correlation heatmap...")
            plt.figure(figsize=(12, 10))
            correlation_matrix = self.df[numerical_cols].corr()
            
            sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('Correlation Matrix - Numerical Features', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
            print(f"  Saved: correlation_heatmap.png")
            plt.close()
            
            # Print strong correlations
            print("\nStrong Correlations (|correlation| > 0.5):")
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if abs(correlation_matrix.iloc[i, j]) > 0.5:
                        print(f"  {correlation_matrix.columns[i]} <-> {correlation_matrix.columns[j]}: {correlation_matrix.iloc[i, j]:.3f}")
        
        # Target variable analysis
        if 'visa_status' in self.df.columns:
            print("\nGenerating visa status analysis...")
            
            # Visa status distribution
            plt.figure(figsize=(8, 6))
            status_counts = self.df['visa_status'].value_counts()
            plt.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', 
                   startangle=90, colors=['#2ecc71', '#e74c3c'])
            plt.title('Visa Status Distribution', fontsize=14, fontweight='bold')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/visa_status_distribution.png', dpi=300, bbox_inches='tight')
            print(f"  Saved: visa_status_distribution.png")
            plt.close()
            
            # Numerical features vs Target
            if numerical_cols:
                n_cols = 3
                n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
                axes = axes.flatten() if n_rows > 1 else [axes]
                
                for idx, col in enumerate(numerical_cols):
                    ax = axes[idx]
                    for status in self.df['visa_status'].unique():
                        data = self.df[self.df['visa_status'] == status][col]
                        ax.hist(data, alpha=0.6, label=status, bins=20, edgecolor='black')
                    ax.set_title(f'{col} by Visa Status', fontsize=10, fontweight='bold')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Frequency')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                # Hide unused subplots
                for idx in range(len(numerical_cols), len(axes)):
                    axes[idx].set_visible(False)
                
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/features_vs_visa_status.png', dpi=300, bbox_inches='tight')
                print(f"  Saved: features_vs_visa_status.png")
                plt.close()
        
        print("\nBivariate analysis completed.\n")
    
    def multivariate_analysis(self):
        """Perform multivariate analysis"""
        print("=" * 80)
        print("3. MULTIVARIATE ANALYSIS")
        print("=" * 80)
        
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Pairplot for key numerical features
        if len(numerical_cols) >= 3:
            print("\nGenerating pairplot for top numerical features...")
            
            # Select top numerical features (limit to 4 for clarity)
            top_features = numerical_cols[:4]
            
            if 'visa_status' in self.df.columns:
                pairplot_data = self.df[top_features + ['visa_status']]
                sns.pairplot(pairplot_data, hue='visa_status', diag_kind='hist', 
                           plot_kws={'alpha': 0.6}, height=2.5)
            else:
                pairplot_data = self.df[top_features]
                sns.pairplot(pairplot_data, diag_kind='hist', height=2.5)
            
            plt.suptitle('Pairplot - Feature Relationships', y=1.02, fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/pairplot_analysis.png', dpi=300, bbox_inches='tight')
            print(f"  Saved: pairplot_analysis.png")
            plt.close()
        
        # Box plots for outlier detection
        if numerical_cols:
            print("\nGenerating box plots for outlier detection...")
            
            n_cols = 3
            n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes]
            
            for idx, col in enumerate(numerical_cols):
                ax = axes[idx]
                self.df.boxplot(column=col, ax=ax)
                ax.set_title(f'Box Plot: {col}', fontsize=10, fontweight='bold')
                ax.set_ylabel(col)
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for idx in range(len(numerical_cols), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/boxplots_outliers.png', dpi=300, bbox_inches='tight')
            print(f"  Saved: boxplots_outliers.png")
            plt.close()
        
        print("\nMultivariate analysis completed.\n")
    
    def feature_engineering(self):
        """Perform feature engineering"""
        print("=" * 80)
        print("4. FEATURE ENGINEERING")
        print("=" * 80)
        
        # Create a copy for feature engineering
        df_engineered = self.df.copy()
        
        print("\nEngineering new features...")
        
        # 1. Age Groups
        if 'age' in df_engineered.columns:
            df_engineered['age_group'] = pd.cut(df_engineered['age'], 
                                                bins=[0, 25, 35, 45, 55, 100],
                                                labels=['18-25', '26-35', '36-45', '46-55', '55+'])
            print("  Created: age_group")
        
        # 2. Work Experience Categories
        if 'work_experience' in df_engineered.columns:
            df_engineered['experience_level'] = pd.cut(df_engineered['work_experience'],
                                                       bins=[-1, 2, 5, 10, 100],
                                                       labels=['Entry', 'Mid', 'Senior', 'Expert'])
            print("  Created: experience_level")
        
        # 3. Language Score Categories
        if 'language_score' in df_engineered.columns:
            df_engineered['language_proficiency'] = pd.cut(df_engineered['language_score'],
                                                           bins=[0, 60, 75, 90, 100],
                                                           labels=['Low', 'Medium', 'High', 'Expert'])
            print("  Created: language_proficiency")
        
        # 4. Financial Status Binary
        if 'financial_status' in df_engineered.columns:
            median_financial = df_engineered['financial_status'].median()
            df_engineered['high_financial_status'] = (df_engineered['financial_status'] >= median_financial).astype(int)
            print("  Created: high_financial_status")
        
        # 5. Total Experience Score
        if 'work_experience' in df_engineered.columns and 'previous_visas' in df_engineered.columns:
            df_engineered['total_experience_score'] = df_engineered['work_experience'] + (df_engineered['previous_visas'] * 2)
            print("  Created: total_experience_score")
        
        # 6. Application Strength Score
        if 'language_score' in df_engineered.columns and 'financial_status' in df_engineered.columns:
            df_engineered['application_strength'] = (
                (df_engineered['language_score'] / 100) * 0.3 +
                (df_engineered['financial_status'] / df_engineered['financial_status'].max()) * 0.3 +
                (df_engineered['work_experience'] / df_engineered['work_experience'].max()) * 0.4
            )
            print("  Created: application_strength")
        
        # 7. Has Support (Job offer or Sponsor)
        if 'job_offer' in df_engineered.columns and 'sponsor' in df_engineered.columns:
            df_engineered['has_support'] = ((df_engineered['job_offer'] == 'Yes') | 
                                           (df_engineered['sponsor'] == 'Yes')).astype(int)
            print("  Created: has_support")
        
        # 8. Dependents Binary
        if 'dependents' in df_engineered.columns:
            df_engineered['has_dependents'] = (df_engineered['dependents'] > 0).astype(int)
            print("  Created: has_dependents")
        
        # 9. Processing Time Categories
        if 'processing_time_days' in df_engineered.columns:
            df_engineered['processing_speed'] = pd.cut(df_engineered['processing_time_days'],
                                                       bins=[0, 30, 60, 90, 1000],
                                                       labels=['Fast', 'Normal', 'Slow', 'Very Slow'])
            print("  Created: processing_speed")
        
        # 10. Age-Experience Ratio
        if 'age' in df_engineered.columns and 'work_experience' in df_engineered.columns:
            df_engineered['age_experience_ratio'] = df_engineered['age'] / (df_engineered['work_experience'] + 1)
            print("  Created: age_experience_ratio")
        
        print(f"\nOriginal features: {len(self.df.columns)}")
        print(f"Engineered features: {len(df_engineered.columns)}")
        print(f"New features added: {len(df_engineered.columns) - len(self.df.columns)}")
        
        # Save engineered dataset
        output_path = '../data/visa_data_engineered.csv'
        df_engineered.to_csv(output_path, index=False)
        print(f"\nEngineered dataset saved to: {output_path}")
        
        # Feature importance analysis for new features
        print("\nAnalyzing engineered features...")
        
        new_numerical_features = ['total_experience_score', 'application_strength', 
                                 'age_experience_ratio']
        
        if all(f in df_engineered.columns for f in new_numerical_features):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            for idx, feature in enumerate(new_numerical_features):
                ax = axes[idx]
                df_engineered[feature].hist(bins=30, edgecolor='black', ax=ax)
                ax.set_title(f'Distribution: {feature}', fontsize=10, fontweight='bold')
                ax.set_xlabel(feature)
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/engineered_features.png', dpi=300, bbox_inches='tight')
            print(f"  Saved: engineered_features.png")
            plt.close()
        
        print("\nFeature engineering completed.\n")
        
        return df_engineered
    
    def generate_eda_summary(self):
        """Generate EDA summary report"""
        print("=" * 80)
        print("EDA SUMMARY REPORT")
        print("=" * 80)
        
        print(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nDataset: {self.data_path}")
        
        # Data quality metrics
        print("\nData Quality Metrics:")
        print(f"  Total Records: {len(self.df)}")
        print(f"  Total Features: {len(self.df.columns)}")
        print(f"  Duplicate Rows: {self.df.duplicated().sum()}")
        print(f"  Missing Values: {self.df.isnull().sum().sum()}")
        
        # Feature breakdown
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        print(f"\nFeature Breakdown:")
        print(f"  Numerical Features: {len(numerical_cols)}")
        print(f"  Categorical Features: {len(categorical_cols)}")
        
        # Target variable distribution
        if 'visa_status' in self.df.columns:
            print(f"\nTarget Variable Distribution:")
            for status, count in self.df['visa_status'].value_counts().items():
                percentage = (count / len(self.df)) * 100
                print(f"  {status}: {count} ({percentage:.2f}%)")
        
        print("\nVisualizations Generated:")
        viz_files = [
            'numerical_distributions.png',
            'categorical_distributions.png',
            'correlation_heatmap.png',
            'visa_status_distribution.png',
            'features_vs_visa_status.png',
            'pairplot_analysis.png',
            'boxplots_outliers.png',
            'engineered_features.png'
        ]
        for viz_file in viz_files:
            if os.path.exists(f'{self.output_dir}/{viz_file}'):
                print(f"  ✓ {viz_file}")
        
        print("\n" + "=" * 80)
        print("MILESTONE 2 EDA & FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print()
    
    def run_complete_eda(self):
        """Execute complete Milestone 2 EDA and Feature Engineering"""
        self.load_data()
        self.univariate_analysis()
        self.bivariate_analysis()
        self.multivariate_analysis()
        df_engineered = self.feature_engineering()
        self.generate_eda_summary()
        
        return df_engineered


def main():
    """Main execution function"""
    # Path to dataset
    data_path = '../data/visa_data.csv'
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return
    
    # Initialize and run EDA
    eda = EDAFeatureEngineering(data_path)
    df_engineered = eda.run_complete_eda()
    
    print("EDA and Feature Engineering completed successfully!")
    print(f"Check the 'visualizations' folder for all generated plots.")
    print(f"Engineered dataset saved as 'visa_data_engineered.csv'")


if __name__ == "__main__":
    main()
