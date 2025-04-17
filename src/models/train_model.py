"""
Model training module for IHD diagnosis project.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.multioutput import ClassifierChain


def get_base_models():
    """
    Get a dictionary of base models.
    
    Returns:
    --------
    dict
        Dictionary of base models.
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'k-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'Ridge Classifier': RidgeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        'Extra Trees Classifier': ExtraTreesClassifier(random_state=42),
        'Classifier Chain': ClassifierChain(LogisticRegression(random_state=42), random_state=42)
    }
    
    return models


def get_param_grids():
    """
    Get parameter grids for hyperparameter tuning.
    
    Returns:
    --------
    dict
        Dictionary of parameter grids.
    """
    param_grids = {
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga'],
            'penalty': ['l1', 'l2']
        },
        'k-Nearest Neighbors': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        'Naive Bayes': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        },
        'Decision Tree': {
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Support Vector Machine': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'linear']
        },
        'Ridge Classifier': {
            'alpha': [0.1, 1.0, 10.0],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Quadratic Discriminant Analysis': {
            'reg_param': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        },
        'AdaBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        },
        'Linear Discriminant Analysis': {
            'solver': ['svd', 'lsqr', 'eigen'],
            'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]
        },
        'Extra Trees Classifier': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Classifier Chain': {
            'base_estimator__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'base_estimator__solver': ['liblinear', 'saga'],
            'base_estimator__penalty': ['l1', 'l2']
        }
    }
    
    return param_grids


def train_model_with_grid_search(model, param_grid, X_train, y_train, cv=5, scoring='accuracy'):
    """
    Train a model with grid search for hyperparameter tuning.
    
    Parameters:
    -----------
    model : estimator
        The model to train.
    param_grid : dict
        Parameter grid for hyperparameter tuning.
    X_train : array-like
        Training features.
    y_train : array-like
        Training target.
    cv : int, optional
        Number of cross-validation folds, by default 5.
    scoring : str, optional
        Scoring metric, by default 'accuracy'.
        
    Returns:
    --------
    GridSearchCV
        Trained model with best parameters.
    """
    # Set up cross-validation
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Set up grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Print best parameters
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best {scoring}: {grid_search.best_score_:.4f}")
    
    return grid_search


def train_all_models(X_train, y_train, models_to_train=None, tune_hyperparameters=True):
    """
    Train all models.
    
    Parameters:
    -----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training target.
    models_to_train : list, optional
        List of model names to train, by default None (train all).
    tune_hyperparameters : bool, optional
        Whether to tune hyperparameters, by default True.
        
    Returns:
    --------
    dict
        Dictionary of trained models.
    """
    # Get base models
    base_models = get_base_models()
    
    # Filter models if specified
    if models_to_train is not None:
        base_models = {k: v for k, v in base_models.items() if k in models_to_train}
    
    # Train models
    trained_models = {}
    
    for name, model in base_models.items():
        print(f"\nTraining {name}...")
        
        if tune_hyperparameters:
            # Get parameter grid
            param_grids = get_param_grids()
            param_grid = param_grids.get(name, {})
            
            # Train with grid search
            if param_grid:
                trained_model = train_model_with_grid_search(model, param_grid, X_train, y_train)
                trained_models[name] = trained_model
            else:
                print(f"No parameter grid defined for {name}. Training without hyperparameter tuning.")
                model.fit(X_train, y_train)
                trained_models[name] = model
        else:
            # Train without hyperparameter tuning
            model.fit(X_train, y_train)
            trained_models[name] = model
    
    return trained_models


def create_voting_ensemble(trained_models, voting='soft'):
    """
    Create a voting ensemble from trained models.
    
    Parameters:
    -----------
    trained_models : dict
        Dictionary of trained models.
    voting : str, optional
        Voting strategy, by default 'soft'.
        
    Returns:
    --------
    VotingClassifier
        Voting ensemble.
    """
    # Extract best estimators from grid search results
    estimators = []
    for name, model in trained_models.items():
        if hasattr(model, 'best_estimator_'):
            estimators.append((name, model.best_estimator_))
        else:
            estimators.append((name, model))
    
    # Create voting ensemble
    ensemble = VotingClassifier(estimators=estimators, voting=voting)
    
    return ensemble


def train_model(X_train, y_train, models_to_train=None, tune_hyperparameters=True, create_ensemble=True):
    """
    Train models and optionally create an ensemble.
    
    Parameters:
    -----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training target.
    models_to_train : list, optional
        List of model names to train, by default None (train all).
    tune_hyperparameters : bool, optional
        Whether to tune hyperparameters, by default True.
    create_ensemble : bool, optional
        Whether to create an ensemble, by default True.
        
    Returns:
    --------
    tuple
        Trained models and ensemble.
    """
    # Train individual models
    trained_models = train_all_models(X_train, y_train, models_to_train, tune_hyperparameters)
    
    # Create ensemble if requested
    ensemble = None
    if create_ensemble and len(trained_models) > 1:
        print("\nCreating voting ensemble...")
        ensemble = create_voting_ensemble(trained_models)
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
    
    return trained_models, ensemble


def save_models(models, ensemble=None, models_dir='models'):
    """
    Save trained models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models.
    ensemble : estimator, optional
        Ensemble model, by default None.
    models_dir : str, optional
        Directory to save models, by default 'models'.
        
    Returns:
    --------
    None
    """
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Save individual models
    for name, model in models.items():
        # Replace spaces with underscores in the filename
        filename = os.path.join(models_dir, f"{name.replace(' ', '_')}.joblib")
        joblib.dump(model, filename)
        print(f"Saved {name} to {filename}")
    
    # Save ensemble if provided
    if ensemble is not None:
        filename = os.path.join(models_dir, "voting_ensemble.joblib")
        joblib.dump(ensemble, filename)
        print(f"Saved ensemble to {filename}")


def load_models(model_names=None, models_dir='models'):
    """
    Load trained models.
    
    Parameters:
    -----------
    model_names : list, optional
        List of model names to load, by default None (load all).
    models_dir : str, optional
        Directory to load models from, by default 'models'.
        
    Returns:
    --------
    dict
        Dictionary of loaded models.
    """
    import os
    
    # Get all model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    
    # Filter model files if model_names is provided
    if model_names is not None:
        model_files = [f for f in model_files if any(name.replace(' ', '_') in f for name in model_names)]
    
    # Load models
    loaded_models = {}
    for model_file in model_files:
        # Get model name from filename
        model_name = os.path.splitext(model_file)[0].replace('_', ' ')
        
        # Load model
        model_path = os.path.join(models_dir, model_file)
        model = joblib.load(model_path)
        loaded_models[model_name] = model
        print(f"Loaded {model_name} from {model_path}")
    
    return loaded_models
