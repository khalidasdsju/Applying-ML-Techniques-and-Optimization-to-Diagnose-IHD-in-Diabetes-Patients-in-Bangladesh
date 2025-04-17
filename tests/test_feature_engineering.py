"""
Unit tests for feature engineering module.
"""

import unittest
import pandas as pd
import numpy as np
from src.data.feature_engineering import (
    calculate_bmi, calculate_age_related_features, calculate_bp_related_features,
    select_features_rfe, select_features_mutual_info, apply_smote
)


class TestFeatureEngineering(unittest.TestCase):
    """
    Test feature engineering functions.
    """
    
    def setUp(self):
        """
        Set up test data.
        """
        # Create test data
        self.data = pd.DataFrame({
            'Age (Years)': [30, 40, 50, 60, 70],
            'Sex (Male/Female)': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'Height (cm)': [170, 160, 180, 165, 175],
            'Weight (kg)': [70, 60, 80, 65, 75],
            'Systolic Blood Pressure (mmHg)': [120, 130, 140, 150, 160],
            'Diastolic Blood Pressure (mmHg)': [80, 85, 90, 95, 100],
            'Random Blood Sugar (mg/dL)': [100, 110, 120, 130, 140],
            'Smoking Status': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'Hypertension (HTN) Status': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'Diabetes Mellitus (DM) Status': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'Ischemic Heart Disease (IHD) Status': [1, 0, 1, 0, 1]
        })
    
    def test_calculate_bmi(self):
        """
        Test calculate_bmi function.
        """
        # Calculate BMI
        data_with_bmi = calculate_bmi(self.data)
        
        # Check if BMI is calculated
        self.assertTrue('BMI' in data_with_bmi.columns)
        
        # Check BMI calculation
        height_m = self.data['Height (cm)'] / 100
        expected_bmi = self.data['Weight (kg)'] / (height_m ** 2)
        pd.testing.assert_series_equal(data_with_bmi['BMI'], expected_bmi)
    
    def test_calculate_age_related_features(self):
        """
        Test calculate_age_related_features function.
        """
        # Calculate age-related features
        data_with_age_features = calculate_age_related_features(self.data)
        
        # Check if age-related features are calculated
        self.assertTrue('Age Group Numeric' in data_with_age_features.columns)
        self.assertTrue('Age Group <30' in data_with_age_features.columns)
        self.assertTrue('Age Group 30-40' in data_with_age_features.columns)
        self.assertTrue('Age Group 40-50' in data_with_age_features.columns)
        self.assertTrue('Age Group 50-60' in data_with_age_features.columns)
        self.assertTrue('Age Group 60-70' in data_with_age_features.columns)
        self.assertTrue('Age Group >70' in data_with_age_features.columns)
    
    def test_calculate_bp_related_features(self):
        """
        Test calculate_bp_related_features function.
        """
        # Calculate blood pressure related features
        data_with_bp_features = calculate_bp_related_features(self.data)
        
        # Check if blood pressure related features are calculated
        self.assertTrue('Pulse Pressure' in data_with_bp_features.columns)
        self.assertTrue('Mean Arterial Pressure' in data_with_bp_features.columns)
        
        # Check pulse pressure calculation
        expected_pulse_pressure = self.data['Systolic Blood Pressure (mmHg)'] - self.data['Diastolic Blood Pressure (mmHg)']
        pd.testing.assert_series_equal(data_with_bp_features['Pulse Pressure'], expected_pulse_pressure)
        
        # Check mean arterial pressure calculation
        expected_map = (
            self.data['Diastolic Blood Pressure (mmHg)'] + 
            (1/3) * (self.data['Systolic Blood Pressure (mmHg)'] - self.data['Diastolic Blood Pressure (mmHg)'])
        )
        pd.testing.assert_series_equal(data_with_bp_features['Mean Arterial Pressure'], expected_map)
    
    def test_select_features_rfe(self):
        """
        Test select_features_rfe function.
        """
        # Prepare data
        X = self.data.drop(columns=['Ischemic Heart Disease (IHD) Status'])
        y = self.data['Ischemic Heart Disease (IHD) Status']
        
        # Encode categorical features
        X_encoded = pd.get_dummies(X)
        
        # Select features
        selected_features = select_features_rfe(X_encoded, y, n_features_to_select=3)
        
        # Check if features are selected
        self.assertEqual(len(selected_features), 3)
        self.assertTrue(all(feature in X_encoded.columns for feature in selected_features))
    
    def test_select_features_mutual_info(self):
        """
        Test select_features_mutual_info function.
        """
        # Prepare data
        X = self.data.drop(columns=['Ischemic Heart Disease (IHD) Status'])
        y = self.data['Ischemic Heart Disease (IHD) Status']
        
        # Encode categorical features
        X_encoded = pd.get_dummies(X)
        
        # Select features
        selected_features = select_features_mutual_info(X_encoded, y, n_features_to_select=3)
        
        # Check if features are selected
        self.assertEqual(len(selected_features), 3)
        self.assertTrue(all(feature in X_encoded.columns for feature in selected_features))
    
    def test_apply_smote(self):
        """
        Test apply_smote function.
        """
        # Prepare data
        X = self.data.drop(columns=['Ischemic Heart Disease (IHD) Status'])
        y = self.data['Ischemic Heart Disease (IHD) Status']
        
        # Encode categorical features
        X_encoded = pd.get_dummies(X)
        
        # Apply SMOTE
        X_resampled, y_resampled = apply_smote(X_encoded, y)
        
        # Check if data is resampled
        self.assertEqual(len(X_resampled), len(y_resampled))
        self.assertTrue(len(X_resampled) >= len(X_encoded))
        
        # Check if class distribution is balanced
        class_counts = pd.Series(y_resampled).value_counts()
        self.assertEqual(class_counts[0], class_counts[1])


if __name__ == '__main__':
    unittest.main()
