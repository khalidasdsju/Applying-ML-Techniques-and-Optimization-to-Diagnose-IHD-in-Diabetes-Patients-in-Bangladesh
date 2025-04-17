"""
Feature engineering module for IHD diagnosis project.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


def calculate_bmi(data):
    """
    Calculate BMI from height and weight.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data with height and weight columns.
        
    Returns:
    --------
    pd.DataFrame
        Data with BMI column added.
    """
    if 'Height (cm)' in data.columns and 'Weight (kg)' in data.columns:
        # Convert height from cm to m
        height_m = data['Height (cm)'] / 100
        # Calculate BMI
        data['BMI'] = data['Weight (kg)'] / (height_m ** 2)
        print("BMI calculated and added to the dataset.")
    else:
        print("Height or weight columns not found. BMI not calculated.")
    
    return data


def calculate_age_related_features(data):
    """
    Calculate age-related features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data with age column.
        
    Returns:
    --------
    pd.DataFrame
        Data with age-related features added.
    """
    if 'Age (Years)' in data.columns:
        # Create age groups
        bins = [0, 30, 40, 50, 60, 70, 100]
        labels = ['<30', '30-40', '40-50', '50-60', '60-70', '>70']
        data['Age Group Numeric'] = pd.cut(data['Age (Years)'], bins=bins, labels=labels, right=False).astype(str)
        
        # Create binary features for age groups
        for label in labels:
            data[f'Age Group {label}'] = (data['Age Group Numeric'] == label).astype(int)
        
        print("Age-related features calculated and added to the dataset.")
    else:
        print("Age column not found. Age-related features not calculated.")
    
    return data


def calculate_bp_related_features(data):
    """
    Calculate blood pressure related features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data with systolic and diastolic blood pressure columns.
        
    Returns:
    --------
    pd.DataFrame
        Data with blood pressure related features added.
    """
    if 'Systolic Blood Pressure (mmHg)' in data.columns and 'Diastolic Blood Pressure (mmHg)' in data.columns:
        # Calculate pulse pressure
        data['Pulse Pressure'] = data['Systolic Blood Pressure (mmHg)'] - data['Diastolic Blood Pressure (mmHg)']
        
        # Calculate mean arterial pressure
        data['Mean Arterial Pressure'] = (
            data['Diastolic Blood Pressure (mmHg)'] + 
            (1/3) * (data['Systolic Blood Pressure (mmHg)'] - data['Diastolic Blood Pressure (mmHg)'])
        )
        
        print("Blood pressure related features calculated and added to the dataset.")
    else:
        print("Blood pressure columns not found. Blood pressure related features not calculated.")
    
    return data


def select_features_rfe(X, y, n_features_to_select=10):
    """
    Select features using Recursive Feature Elimination.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    n_features_to_select : int, optional
        Number of features to select, by default 10.
        
    Returns:
    --------
    list
        Selected feature names.
    """
    # Initialize the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Initialize RFE
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    
    # Fit RFE
    rfe.fit(X, y)
    
    # Get selected features
    selected_features = X.columns[rfe.support_].tolist()
    
    print(f"Selected {len(selected_features)} features using RFE.")
    return selected_features


def select_features_mutual_info(X, y, n_features_to_select=10):
    """
    Select features using mutual information.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    n_features_to_select : int, optional
        Number of features to select, by default 10.
        
    Returns:
    --------
    list
        Selected feature names.
    """
    # Calculate mutual information
    mi_scores = mutual_info_classif(X, y)
    
    # Create a dataframe of features and their scores
    mi_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
    
    # Sort by score
    mi_df = mi_df.sort_values('MI Score', ascending=False)
    
    # Select top features
    selected_features = mi_df.head(n_features_to_select)['Feature'].tolist()
    
    print(f"Selected {len(selected_features)} features using mutual information.")
    return selected_features


def apply_smote(X, y, random_state=42):
    """
    Apply SMOTE to handle class imbalance.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    random_state : int, optional
        Random state for reproducibility, by default 42.
        
    Returns:
    --------
    tuple
        X_resampled, y_resampled.
    """
    # Check class distribution
    class_counts = y.value_counts()
    print("Class distribution before SMOTE:")
    print(class_counts)
    
    # Apply SMOTE
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Check class distribution after SMOTE
    resampled_class_counts = pd.Series(y_resampled).value_counts()
    print("Class distribution after SMOTE:")
    print(resampled_class_counts)
    
    return X_resampled, y_resampled


def engineer_features(data, target_column='Ischemic Heart Disease (IHD) Status', apply_feature_selection=True, apply_smote_resampling=True):
    """
    Apply feature engineering to the dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data.
    target_column : str, optional
        Target column name, by default 'Ischemic Heart Disease (IHD) Status'.
    apply_feature_selection : bool, optional
        Whether to apply feature selection, by default True.
    apply_smote_resampling : bool, optional
        Whether to apply SMOTE resampling, by default True.
        
    Returns:
    --------
    tuple
        X, y, selected_features.
    """
    # Calculate derived features
    data = calculate_bmi(data)
    data = calculate_age_related_features(data)
    data = calculate_bp_related_features(data)
    
    # Split features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Apply feature selection if requested
    selected_features = None
    if apply_feature_selection:
        # Combine features from both methods
        rfe_features = select_features_rfe(X, y)
        mi_features = select_features_mutual_info(X, y)
        
        # Get unique features
        selected_features = list(set(rfe_features + mi_features))
        
        # Filter X to include only selected features
        X = X[selected_features]
    
    # Apply SMOTE if requested
    if apply_smote_resampling:
        X, y = apply_smote(X, y)
    
    return X, y, selected_features
