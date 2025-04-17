"""
Data preprocessing module for IHD diagnosis project.
"""

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(file_path):
    """
    Load data from SPSS file.
    
    Parameters:
    -----------
    file_path : str
        Path to the SPSS file.
        
    Returns:
    --------
    pd.DataFrame
        Loaded data.
    """
    try:
        data = pd.read_spss(file_path)
        print(f"Data loaded successfully with shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def clean_column_names(data):
    """
    Clean and standardize column names.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data.
        
    Returns:
    --------
    pd.DataFrame
        Data with cleaned column names.
    """
    # Define new column names
    new_columns = [
        'Age (Years)', 'Sex (Male/Female)', 'Occupation Type', 'Education Level', 'Economic Status',
        'Height (cm)', 'Weight (kg)', 'Systolic Blood Pressure (mmHg)', 'Diastolic Blood Pressure (mmHg)',
        'Random Blood Sugar (mg/dL)', 'Smoking Status', 'Hypertension (HTN) Status',
        'Diabetes Mellitus (DM) Status', 'Dyslipidemia Status', 'Stroke Status', 'Ischemic Heart Disease (IHD) Status',
        'Age Group', 'Body Mass Index (BMI) Group', 'Hypertension Stage'
    ]
    
    # Check if the lengths match
    if len(data.columns) == len(new_columns):
        data.columns = new_columns
        return data
    else:
        print(f"Column length mismatch: {len(data.columns)} vs {len(new_columns)}")
        return data


def handle_missing_values(data):
    """
    Handle missing values in the dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data.
        
    Returns:
    --------
    pd.DataFrame
        Data with handled missing values.
    """
    # Check for missing values
    missing_values = data.isnull().sum()
    print("Missing Values in Each Column:")
    print(missing_values[missing_values > 0])
    
    # Fill missing values
    if 'Random Blood Sugar (mg/dL)' in data.columns:
        data['Random Blood Sugar (mg/dL)'].fillna(data['Random Blood Sugar (mg/dL)'].median(), inplace=True)
    
    return data


def detect_outliers_iqr(data):
    """
    Detect outliers using IQR method.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data.
        
    Returns:
    --------
    pd.Series
        Count of outliers in each numerical column.
    """
    numerical_data = data.select_dtypes(include=[np.number])
    Q1 = numerical_data.quantile(0.25)
    Q3 = numerical_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = ((numerical_data < lower_bound) | (numerical_data > upper_bound)).sum()
    return outliers


def handle_outliers(data, max_iterations=5):
    """
    Handle outliers using iterative median imputation and winsorization.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data.
    max_iterations : int, optional
        Maximum number of iterations for outlier handling, by default 5.
        
    Returns:
    --------
    pd.DataFrame
        Data with handled outliers.
    """
    # First, try iterative median imputation
    numerical_data = data.select_dtypes(include=[np.number])
    for iteration in range(max_iterations):
        Q1 = numerical_data.quantile(0.25)
        Q3 = numerical_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify and replace outliers with the median for each column
        for column in numerical_data.columns:
            outliers = (numerical_data[column] < lower_bound[column]) | (numerical_data[column] > upper_bound[column])
            if outliers.sum() > 0:
                median_value = numerical_data[column].median()
                numerical_data[column] = numerical_data[column].where(~outliers, median_value)
        
        # After replacing outliers, check if there are any outliers left
        outliers_after = detect_outliers_iqr(numerical_data)
        if all(outliers_after == 0):
            print(f"Outliers have been handled after {iteration + 1} iterations.")
            break
    
    # Apply Winsorization for any remaining outliers
    for column in numerical_data.columns:
        if detect_outliers_iqr(numerical_data)[column] > 0:
            numerical_data[column] = winsorize(numerical_data[column], limits=(0.05, 0.05))
    
    # Update the original dataframe
    data[numerical_data.columns] = numerical_data
    return data


def encode_categorical_features(data):
    """
    Encode categorical features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data.
        
    Returns:
    --------
    pd.DataFrame
        Data with encoded categorical features.
    dict
        Dictionary of label encoders.
    """
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    
    return data, label_encoders


def scale_numerical_features(data):
    """
    Scale numerical features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data.
        
    Returns:
    --------
    pd.DataFrame
        Data with scaled numerical features.
    StandardScaler
        Fitted scaler.
    """
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    return data, scaler


def preprocess_data(file_path):
    """
    Preprocess data for model training.
    
    Parameters:
    -----------
    file_path : str
        Path to the data file.
        
    Returns:
    --------
    tuple
        Preprocessed data, label encoders, and scaler.
    """
    # Load data
    data = load_data(file_path)
    if data is None:
        return None, None, None
    
    # Clean column names
    data = clean_column_names(data)
    
    # Drop unnecessary columns if they exist
    if 'Serial_No' in data.columns:
        data = data.drop(columns=['Serial_No'])
    if 'Name' in data.columns:
        data = data.drop(columns=['Name'])
    
    # Handle missing values
    data = handle_missing_values(data)
    
    # Handle outliers
    data = handle_outliers(data)
    
    # Encode categorical features
    data, label_encoders = encode_categorical_features(data)
    
    # Scale numerical features
    data, scaler = scale_numerical_features(data)
    
    return data, label_encoders, scaler


def split_data(data, target_column='Ischemic Heart Disease (IHD) Status', test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data.
    target_column : str, optional
        Target column name, by default 'Ischemic Heart Disease (IHD) Status'.
    test_size : float, optional
        Test set size, by default 0.2.
    random_state : int, optional
        Random state for reproducibility, by default 42.
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test.
    """
    from sklearn.model_selection import train_test_split
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    return X_train, X_test, y_train, y_test
