"""
Unit tests for preprocessing module.
"""

import unittest
import pandas as pd
import numpy as np
from src.data.preprocessing import (
    clean_column_names, handle_missing_values, detect_outliers_iqr,
    handle_outliers, encode_categorical_features, scale_numerical_features
)


class TestPreprocessing(unittest.TestCase):
    """
    Test preprocessing functions.
    """
    
    def setUp(self):
        """
        Set up test data.
        """
        # Create test data
        self.data = pd.DataFrame({
            'Age': [30, 40, 50, 60, 70],
            'Sex': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'Height': [170, 160, 180, 165, 175],
            'Weight': [70, 60, 80, 65, 75],
            'Systolic': [120, 130, 140, 150, 160],
            'Diastolic': [80, 85, 90, 95, 100],
            'RBS': [100, np.nan, 120, 130, 140],
            'Smoking': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'HTN': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'DM': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'IHD': ['Yes', 'No', 'Yes', 'No', 'Yes']
        })
    
    def test_clean_column_names(self):
        """
        Test clean_column_names function.
        """
        # Define new column names
        new_columns = [
            'Age (Years)', 'Sex (Male/Female)', 'Height (cm)', 'Weight (kg)',
            'Systolic Blood Pressure (mmHg)', 'Diastolic Blood Pressure (mmHg)',
            'Random Blood Sugar (mg/dL)', 'Smoking Status', 'Hypertension (HTN) Status',
            'Diabetes Mellitus (DM) Status', 'Ischemic Heart Disease (IHD) Status'
        ]
        
        # Clean column names
        data_copy = self.data.copy()
        data_copy.columns = new_columns
        
        # Check if column names are cleaned
        self.assertEqual(list(data_copy.columns), new_columns)
    
    def test_handle_missing_values(self):
        """
        Test handle_missing_values function.
        """
        # Handle missing values
        data_copy = self.data.copy()
        data_copy = handle_missing_values(data_copy)
        
        # Check if missing values are handled
        self.assertEqual(data_copy.isnull().sum().sum(), 0)
    
    def test_detect_outliers_iqr(self):
        """
        Test detect_outliers_iqr function.
        """
        # Create data with outliers
        data_with_outliers = self.data.copy()
        data_with_outliers.loc[0, 'Height'] = 300  # Add outlier
        
        # Detect outliers
        outliers = detect_outliers_iqr(data_with_outliers)
        
        # Check if outlier is detected
        self.assertTrue(outliers['Height'] > 0)
    
    def test_handle_outliers(self):
        """
        Test handle_outliers function.
        """
        # Create data with outliers
        data_with_outliers = self.data.copy()
        data_with_outliers.loc[0, 'Height'] = 300  # Add outlier
        
        # Handle outliers
        data_without_outliers = handle_outliers(data_with_outliers)
        
        # Detect outliers after handling
        outliers = detect_outliers_iqr(data_without_outliers)
        
        # Check if outliers are handled
        self.assertTrue(all(outliers == 0))
    
    def test_encode_categorical_features(self):
        """
        Test encode_categorical_features function.
        """
        # Encode categorical features
        data_copy = self.data.copy()
        encoded_data, label_encoders = encode_categorical_features(data_copy)
        
        # Check if categorical features are encoded
        self.assertTrue(all(encoded_data.dtypes != 'object'))
        self.assertTrue(all(encoded_data.dtypes != 'category'))
        
        # Check if label encoders are created
        self.assertTrue('Sex' in label_encoders)
        self.assertTrue('Smoking' in label_encoders)
        self.assertTrue('HTN' in label_encoders)
        self.assertTrue('DM' in label_encoders)
        self.assertTrue('IHD' in label_encoders)
    
    def test_scale_numerical_features(self):
        """
        Test scale_numerical_features function.
        """
        # Scale numerical features
        data_copy = self.data.copy()
        scaled_data, scaler = scale_numerical_features(data_copy)
        
        # Check if numerical features are scaled
        self.assertTrue(abs(scaled_data['Age'].mean()) < 1e-10)
        self.assertTrue(abs(scaled_data['Height'].mean()) < 1e-10)
        self.assertTrue(abs(scaled_data['Weight'].mean()) < 1e-10)
        self.assertTrue(abs(scaled_data['Systolic'].mean()) < 1e-10)
        self.assertTrue(abs(scaled_data['Diastolic'].mean()) < 1e-10)
        
        # Check if scaler is created
        self.assertIsNotNone(scaler)


if __name__ == '__main__':
    unittest.main()
