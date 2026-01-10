"""
Prediction Service - Handles visa status and processing time predictions
"""

import joblib
import pandas as pd
import numpy as np
from typing import Dict, Tuple


class VisaPredictor:
    """Visa status and processing time predictor"""
    
    def __init__(self, models_dir='../models'):
        """Initialize the predictor with trained models"""
        self.models_dir = models_dir
        self.load_models()
    
    def load_models(self):
        """Load all trained models and encoders"""
        try:
            self.status_classifier = joblib.load(f'{self.models_dir}/visa_status_classifier.pkl')
            self.time_regressor = joblib.load(f'{self.models_dir}/processing_time_regressor.pkl')
            self.status_encoder = joblib.load(f'{self.models_dir}/status_label_encoder.pkl')
            self.categorical_encoders = joblib.load(f'{self.models_dir}/categorical_encoders.pkl')
            
            # Load feature importance for explanations
            self.feature_importance = pd.read_csv(f'{self.models_dir}/feature_importance.csv')
            
            print("✅ Models loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def prepare_input(self, application_data: Dict) -> pd.DataFrame:
        """Prepare input data for prediction"""
        
        # Define all expected features in the correct order
        feature_names = [
            'age', 'country', 'education', 'work_experience', 'language_score',
            'financial_status', 'previous_visas', 'travel_history', 'visa_type',
            'purpose', 'marital_status', 'dependents', 'application_complete',
            'job_offer', 'sponsor'
        ]
        
        # Create DataFrame
        df = pd.DataFrame([application_data])
        
        # Encode categorical features
        categorical_columns = ['country', 'education', 'visa_type', 'purpose', 'marital_status']
        
        for col in categorical_columns:
            if col in df.columns:
                le = self.categorical_encoders[col]
                try:
                    df[col] = le.transform(df[col])
                except ValueError:
                    # Handle unknown categories by using the most common category
                    df[col] = 0
        
        # Ensure all features are present and in correct order
        df = df[feature_names]
        
        return df
    
    def predict_status(self, X: pd.DataFrame) -> Tuple[str, float]:
        """Predict visa status and confidence"""
        
        # Get prediction probabilities
        proba = self.status_classifier.predict_proba(X)[0]
        
        # Get predicted class
        prediction = self.status_classifier.predict(X)[0]
        status = self.status_encoder.inverse_transform([prediction])[0]
        
        # Confidence is the probability of the predicted class
        confidence = float(proba[prediction])
        
        return status, confidence
    
    def predict_processing_time(self, X: pd.DataFrame) -> int:
        """Predict processing time in days"""
        
        time_prediction = self.time_regressor.predict(X)[0]
        
        # Round to nearest integer and ensure positive
        time_days = max(1, int(round(time_prediction)))
        
        return time_days
    
    def get_key_factors(self, X: pd.DataFrame, top_n: int = 5) -> list:
        """Get the most important factors for this prediction"""
        
        # Get feature values
        feature_values = X.iloc[0].to_dict()
        
        # Get top N important features
        top_features = self.feature_importance.head(top_n)
        
        key_factors = []
        for _, row in top_features.iterrows():
            feature = row['feature']
            importance = row['importance']
            value = feature_values.get(feature, 'N/A')
            
            key_factors.append({
                'factor': self._format_feature_name(feature),
                'importance': float(importance),
                'value': str(value)
            })
        
        return key_factors
    
    def _format_feature_name(self, feature: str) -> str:
        """Format feature name for display"""
        name_mapping = {
            'age': 'Age',
            'country': 'Country of Origin',
            'education': 'Education Level',
            'work_experience': 'Work Experience (years)',
            'language_score': 'Language Proficiency Score',
            'financial_status': 'Financial Status ($)',
            'previous_visas': 'Previous Visas',
            'travel_history': 'Travel History (countries)',
            'visa_type': 'Visa Type',
            'purpose': 'Application Purpose',
            'marital_status': 'Marital Status',
            'dependents': 'Number of Dependents',
            'application_complete': 'Application Completeness',
            'job_offer': 'Job Offer Status',
            'sponsor': 'Sponsor Available'
        }
        return name_mapping.get(feature, feature.replace('_', ' ').title())
    
    def predict(self, application_data: Dict) -> Dict:
        """Complete prediction pipeline"""
        
        # Prepare input
        X = self.prepare_input(application_data)
        
        # Predict visa status
        status, confidence = self.predict_status(X)
        
        # Predict processing time
        processing_time = self.predict_processing_time(X)
        
        # Get key factors
        key_factors = self.get_key_factors(X)
        
        # Build response
        result = {
            'visa_status': status,
            'confidence': round(confidence * 100, 2),
            'processing_time_days': processing_time,
            'processing_time_weeks': round(processing_time / 7, 1),
            'key_factors': key_factors,
            'recommendation': self._get_recommendation(status, confidence)
        }
        
        return result
    
    def _get_recommendation(self, status: str, confidence: float) -> str:
        """Generate recommendation based on prediction"""
        
        if status == 'Approved':
            if confidence >= 0.85:
                return "Your application shows strong indicators for approval. Ensure all documents are complete and accurate."
            elif confidence >= 0.70:
                return "Your application has good chances of approval. Consider strengthening weak areas identified in key factors."
            else:
                return "Your application may be approved, but consider improving key factors to increase your chances."
        else:  # Rejected
            if confidence >= 0.85:
                return "Your application shows several weak points. We strongly recommend addressing the key factors before applying."
            elif confidence >= 0.70:
                return "Your application may face challenges. Consider improving the identified key factors significantly."
            else:
                return "The prediction is uncertain. You may want to consult with an immigration specialist before applying."


# Test function
def test_predictor():
    """Test the predictor with sample data"""
    
    predictor = VisaPredictor()
    
    # Test case 1: Strong application
    sample_application = {
        'age': 32,
        'country': 'India',
        'education': 'Master',
        'work_experience': 7,
        'language_score': 7.5,
        'financial_status': 75000,
        'previous_visas': 1,
        'travel_history': 5,
        'visa_type': 'Work',
        'purpose': 'Employment',
        'marital_status': 'Married',
        'dependents': 1,
        'application_complete': 1,
        'job_offer': 1,
        'sponsor': 1
    }
    
    result = predictor.predict(sample_application)
    
    print("\n📊 Test Prediction Result:")
    print(f"  Status: {result['visa_status']}")
    print(f"  Confidence: {result['confidence']}%")
    print(f"  Processing Time: {result['processing_time_days']} days ({result['processing_time_weeks']} weeks)")
    print(f"\n  Key Factors:")
    for factor in result['key_factors']:
        print(f"    - {factor['factor']}: {factor['value']}")
    print(f"\n  💡 Recommendation: {result['recommendation']}")


if __name__ == "__main__":
    test_predictor()
