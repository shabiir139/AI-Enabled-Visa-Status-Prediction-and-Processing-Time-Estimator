"""
Visa Status Prediction and Processing Time Estimation - Model Trainer
This script trains machine learning models for visa prediction
"""

import pandas as pd
import numpy as np
try:
    # When executed as a module inside package
    from . import data_processing as dp
except Exception:
    # When executed as a script, fall back to local import
    import data_processing as dp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib
import os


def load_and_prepare_data(data_path='../data/visa_data.csv'):
    """Load and prepare the dataset"""
    print("📊 Loading dataset...")
    # Run preprocessing pipeline which will save encoders and a processed CSV
    processed_path = '../data/processed_visa_data.csv'
    try:
        df = dp.preprocess_data(input_path=data_path, output_path=processed_path, models_dir='../models')
        print(f"✅ Loaded and processed {len(df)} records from {processed_path}")
        print(f"📋 Features: {list(df.columns)}")
        return df
    except Exception:
        # Fallback: try to load raw CSV if preprocessing fails
        df = pd.read_csv(data_path)
        print(f"⚠️ Preprocessing failed, loaded raw data with {len(df)} records")
        return df


def encode_categorical_features(df):
    """Encode categorical features"""
    print("\n🔄 Encoding categorical features...")
    
    # Create a copy to avoid modifying original
    df_encoded = df.copy()
    
    # Create label encoders dictionary
    label_encoders = {}
    
    categorical_columns = ['country', 'education', 'visa_type', 'purpose', 'marital_status']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
        print(f"  ✓ Encoded {col}: {len(le.classes_)} unique values")
    
    return df_encoded, label_encoders


def train_visa_status_model(X_train, X_test, y_train, y_test):
    """Train the visa status classification model"""
    print("\n🤖 Training Visa Status Prediction Model...")
    
    # Encode target variable
    le_status = LabelEncoder()
    y_train_encoded = le_status.fit_transform(y_train)
    y_test_encoded = le_status.transform(y_test)
    
    # Train Random Forest Classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train_encoded)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test_encoded, y_pred)
    print(f"\n📊 Model Performance:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test_encoded, y_pred, target_names=le_status.classes_))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🎯 Top 10 Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    return clf, le_status, feature_importance


def train_processing_time_model(X_train, X_test, y_train, y_test):
    """Train the processing time estimation model"""
    print("\n⏱️ Training Processing Time Estimation Model...")
    
    # Train Gradient Boosting Regressor
    reg = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    reg.fit(X_train, y_train)
    
    # Make predictions
    y_pred = reg.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 Model Performance:")
    print(f"  RMSE: {rmse:.2f} days")
    print(f"  R² Score: {r2:.2%}")
    print(f"  Mean Absolute Error: {np.mean(np.abs(y_test - y_pred)):.2f} days")
    
    return reg


def save_models(clf, reg, le_status, label_encoders, feature_importance):
    """Save all trained models and encoders"""
    print("\n💾 Saving models...")
    
    # Create models directory if it doesn't exist
    os.makedirs('../models', exist_ok=True)
    
    # Save models
    joblib.dump(clf, '../models/visa_status_classifier.pkl')
    joblib.dump(reg, '../models/processing_time_regressor.pkl')
    joblib.dump(le_status, '../models/status_label_encoder.pkl')
    joblib.dump(label_encoders, '../models/categorical_encoders.pkl')
    
    # Save feature importance
    feature_importance.to_csv('../models/feature_importance.csv', index=False)
    
    print("✅ Models saved successfully!")
    print("  📁 Models directory: ../models/")


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("🚀 VISA STATUS PREDICTION - MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    df = load_and_prepare_data()
    
    # Encode categorical features
    df_encoded, label_encoders = encode_categorical_features(df)
    
    # Prepare features and targets
    print("\n📦 Preparing features and targets...")
    
    # Features (all except visa_status and processing_time_days)
    feature_cols = [col for col in df_encoded.columns 
                   if col not in ['visa_status', 'processing_time_days']]
    X = df_encoded[feature_cols]
    
    # Targets
    y_status = df['visa_status']  # Keep original labels
    y_time = df_encoded['processing_time_days']
    
    # Split data
    X_train, X_test, y_status_train, y_status_test = train_test_split(
        X, y_status, test_size=0.2, random_state=42, stratify=y_status
    )
    
    _, _, y_time_train, y_time_test = train_test_split(
        X, y_time, test_size=0.2, random_state=42
    )
    
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # Train models
    clf, le_status, feature_importance = train_visa_status_model(
        X_train, X_test, y_status_train, y_status_test
    )
    
    reg = train_processing_time_model(
        X_train, X_test, y_time_train, y_time_test
    )
    
    # Save models
    save_models(clf, reg, le_status, label_encoders, feature_importance)
    
    print("\n" + "=" * 60)
    print("✨ Training Complete! Models are ready to use.")
    print("=" * 60)


if __name__ == "__main__":
    main()
