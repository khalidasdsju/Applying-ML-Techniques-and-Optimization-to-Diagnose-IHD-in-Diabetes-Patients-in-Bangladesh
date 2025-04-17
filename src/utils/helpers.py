"""
Helper functions for IHD diagnosis project.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib


def create_directory(directory):
    """
    Create directory if it doesn't exist.
    
    Parameters:
    -----------
    directory : str
        Directory path.
        
    Returns:
    --------
    None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def save_dataframe(df, file_path):
    """
    Save dataframe to CSV file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to save.
    file_path : str
        File path.
        
    Returns:
    --------
    None
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(file_path)
    create_directory(directory)
    
    # Save dataframe
    df.to_csv(file_path, index=False)
    print(f"Saved dataframe to {file_path}")


def load_dataframe(file_path):
    """
    Load dataframe from CSV file.
    
    Parameters:
    -----------
    file_path : str
        File path.
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    # Load dataframe
    df = pd.read_csv(file_path)
    print(f"Loaded dataframe from {file_path} with shape: {df.shape}")
    
    return df


def save_figure(fig, file_path, dpi=300):
    """
    Save figure to file.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save.
    file_path : str
        File path.
    dpi : int, optional
        DPI, by default 300.
        
    Returns:
    --------
    None
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(file_path)
    create_directory(directory)
    
    # Save figure
    fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {file_path}")


def save_model(model, file_path):
    """
    Save model to file.
    
    Parameters:
    -----------
    model : estimator
        Model to save.
    file_path : str
        File path.
        
    Returns:
    --------
    None
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(file_path)
    create_directory(directory)
    
    # Save model
    joblib.dump(model, file_path)
    print(f"Saved model to {file_path}")


def load_model(file_path):
    """
    Load model from file.
    
    Parameters:
    -----------
    file_path : str
        File path.
        
    Returns:
    --------
    estimator
        Loaded model.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    # Load model
    model = joblib.load(file_path)
    print(f"Loaded model from {file_path}")
    
    return model


def get_feature_names(X):
    """
    Get feature names from feature matrix.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix.
        
    Returns:
    --------
    list
        Feature names.
    """
    if hasattr(X, 'columns'):
        return X.columns.tolist()
    elif hasattr(X, 'feature_names_in_'):
        return X.feature_names_in_.tolist()
    else:
        return [f'Feature {i}' for i in range(X.shape[1])]


def print_model_summary(model):
    """
    Print model summary.
    
    Parameters:
    -----------
    model : estimator
        Model to summarize.
        
    Returns:
    --------
    None
    """
    print(f"Model: {type(model).__name__}")
    
    # Print model parameters
    if hasattr(model, 'get_params'):
        params = model.get_params()
        print("\nParameters:")
        for param, value in params.items():
            print(f"  {param}: {value}")
    
    # Print feature importances if available
    if hasattr(model, 'feature_importances_'):
        print("\nFeature Importances:")
        feature_names = get_feature_names(model)
        importances = model.feature_importances_
        
        # Sort importances
        indices = np.argsort(importances)[::-1]
        
        for i in range(min(10, len(feature_names))):
            print(f"  {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Print coefficients if available
    if hasattr(model, 'coef_'):
        print("\nCoefficients:")
        feature_names = get_feature_names(model)
        coefficients = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        
        # Sort coefficients
        indices = np.argsort(np.abs(coefficients))[::-1]
        
        for i in range(min(10, len(feature_names))):
            print(f"  {feature_names[indices[i]]}: {coefficients[indices[i]]:.4f}")


def get_class_distribution(y):
    """
    Get class distribution.
    
    Parameters:
    -----------
    y : array-like
        Target variable.
        
    Returns:
    --------
    pd.Series
        Class distribution.
    """
    # Convert to pandas Series
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    
    # Get class counts
    class_counts = y.value_counts()
    
    # Calculate percentages
    class_percentages = class_counts / class_counts.sum() * 100
    
    # Create distribution dataframe
    distribution = pd.DataFrame({
        'Count': class_counts,
        'Percentage': class_percentages
    })
    
    return distribution


def print_class_distribution(y, title="Class Distribution"):
    """
    Print class distribution.
    
    Parameters:
    -----------
    y : array-like
        Target variable.
    title : str, optional
        Title, by default "Class Distribution".
        
    Returns:
    --------
    None
    """
    # Get class distribution
    distribution = get_class_distribution(y)
    
    # Print distribution
    print(title)
    print(distribution)
    
    # Print total
    print(f"\nTotal: {len(y)}")


def plot_class_distribution(y, figsize=(10, 6), title="Class Distribution"):
    """
    Plot class distribution.
    
    Parameters:
    -----------
    y : array-like
        Target variable.
    figsize : tuple, optional
        Figure size, by default (10, 6).
    title : str, optional
        Title, by default "Class Distribution".
        
    Returns:
    --------
    matplotlib.figure.Figure
        Class distribution plot.
    """
    # Get class distribution
    distribution = get_class_distribution(y)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    bars = ax.bar(
        distribution.index,
        distribution['Count'],
        color='skyblue',
        edgecolor='black',
        alpha=0.7
    )
    
    # Add count and percentage labels
    for bar, count, percentage in zip(bars, distribution['Count'], distribution['Percentage']):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{count} ({percentage:.1f}%)',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    # Set labels
    ax.set_title(title)
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    return fig
