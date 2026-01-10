import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import joblib


def load_data(path='../data/visa_data.csv'):
    print(f"📥 Loading raw data from: {path}")
    df = pd.read_csv(path)
    print(f"  → {len(df)} rows, {len(df.columns)} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: strip strings, fix dtypes, fill obvious missing values."""
    print("🧹 Cleaning data...")

    df_clean = df.copy()

    # Trim whitespace from object columns
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # Convert numeric-like columns to numeric with coercion
    numeric_cols = ['age', 'work_experience', 'language_score', 'financial_status',
                    'previous_visas', 'travel_history', 'dependents', 'processing_time_days']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Fix boolean/integer flags
    flag_cols = ['application_complete', 'job_offer', 'sponsor']
    for col in flag_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0).astype(int)

    # Drop duplicate rows if any
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    after = len(df_clean)
    if before != after:
        print(f"  → Dropped {before-after} duplicate rows")

    return df_clean


def handle_outliers(df: pd.DataFrame, numeric_cols=None) -> pd.DataFrame:
    """Simple IQR-based outlier clipping to reduce extreme values' influence."""
    if numeric_cols is None:
        numeric_cols = ['age', 'work_experience', 'language_score', 'financial_status',
                        'previous_visas', 'travel_history', 'dependents', 'processing_time_days']

    df_out = df.copy()
    for col in numeric_cols:
        if col in df_out.columns:
            q1 = df_out[col].quantile(0.25)
            q3 = df_out[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df_out[col] = df_out[col].clip(lower=lower, upper=upper)

    return df_out


def build_and_save_preprocessors(df: pd.DataFrame, models_dir='../models'):
    """Fit imputers/scalers and label encoders and save them for later use.

    Returns the fitted objects in a dict.
    """
    print("⚙️ Building preprocessors and encoders...")
    os.makedirs(models_dir, exist_ok=True)

    numeric_cols = ['age', 'work_experience', 'language_score', 'financial_status',
                    'previous_visas', 'travel_history', 'dependents']
    categorical_cols = ['country', 'education', 'visa_type', 'purpose', 'marital_status']

    # Numeric pipeline: impute median + scale
    numeric_imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    numeric_pipeline = Pipeline([('imputer', numeric_imputer), ('scaler', scaler)])
    numeric_pipeline.fit(df[numeric_cols])

    # Save numeric pipeline
    joblib.dump(numeric_pipeline, os.path.join(models_dir, 'numeric_preprocessor.pkl'))
    print(f"  → Saved numeric preprocessor to {models_dir}/numeric_preprocessor.pkl")

    # Label encoders for categorical variables (keeps mapping simple for predictor)
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            # Fill missing values before fitting
            values = df[col].fillna('MISSING').astype(str)
            le.fit(values)
            label_encoders[col] = le
            print(f"  ✓ Fitted LabelEncoder for {col} ({len(le.classes_)} classes)")

    # Save label encoders
    joblib.dump(label_encoders, os.path.join(models_dir, 'categorical_encoders.pkl'))
    print(f"  → Saved categorical encoders to {models_dir}/categorical_encoders.pkl")

    return {
        'numeric_pipeline': numeric_pipeline,
        'label_encoders': label_encoders
    }


def preprocess_data(input_path='../data/visa_data.csv', output_path='../data/processed_visa_data.csv', models_dir='../models'):
    """Complete preprocessing pipeline: load, clean, handle outliers, fit and save encoders, and
    produce a processed CSV where categorical columns are label-encoded (useful for quick training runs).
    """
    df = load_data(input_path)
    df = clean_data(df)
    df = handle_outliers(df)

    # Fit and save preprocessors
    fitted = build_and_save_preprocessors(df, models_dir=models_dir)

    # Apply label encoders to categorical columns and save processed CSV
    df_processed = df.copy()
    for col, le in fitted['label_encoders'].items():
        if col in df_processed.columns:
            df_processed[col] = le.transform(df_processed[col].fillna('MISSING').astype(str))

    # Save processed dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    print(f"✅ Processed dataset saved to: {output_path}")

    return df_processed


if __name__ == '__main__':
    print('=' * 60)
    print('🔁 RUNNING DATA PROCESSING PIPELINE')
    print('=' * 60)
    preprocess_data()
    print('\n✨ Data processing complete.')
