"""
Model evaluation module for IHD diagnosis project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, log_loss, precision_recall_curve,
    roc_auc_score, precision_score, recall_score, f1_score
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model on test data.
    
    Parameters:
    -----------
    model : estimator
        Trained model.
    X_test : array-like
        Test features.
    y_test : array-like
        Test target.
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics.
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)
        # For binary classification, we need the probability of the positive class
        if y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]
    else:
        y_prob = None
    
    # Calculate metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
    metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
    metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
    
    # Calculate ROC AUC if probabilities are available
    if y_prob is not None:
        if len(np.unique(y_test)) == 2:  # Binary classification
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        else:  # Multi-class classification
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob, multi_class='ovr')
    
    # Calculate log loss if probabilities are available
    if y_prob is not None:
        metrics['log_loss'] = log_loss(y_test, y_prob)
    
    # Print metrics
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    if 'log_loss' in metrics:
        print(f"Log Loss: {metrics['log_loss']:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return metrics


def plot_confusion_matrix(model, X_test, y_test, class_names=None, figsize=(10, 8)):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    model : estimator
        Trained model.
    X_test : array-like
        Test features.
    y_test : array-like
        Test target.
    class_names : list, optional
        List of class names, by default None.
    figsize : tuple, optional
        Figure size, by default (10, 8).
        
    Returns:
    --------
    matplotlib.figure.Figure
        Confusion matrix plot.
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot confusion matrix
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=class_names, yticklabels=class_names
    )
    
    # Set labels
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    return fig


def plot_roc_curve(model, X_test, y_test, figsize=(10, 8)):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    model : estimator
        Trained model.
    X_test : array-like
        Test features.
    y_test : array-like
        Test target.
    figsize : tuple, optional
        Figure size, by default (10, 8).
        
    Returns:
    --------
    matplotlib.figure.Figure
        ROC curve plot.
    """
    # Check if model can predict probabilities
    if not hasattr(model, 'predict_proba'):
        print("Model does not support predict_proba. Cannot plot ROC curve.")
        return None
    
    # Get probabilities
    y_prob = model.predict_proba(X_test)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # For binary classification
    if y_prob.shape[1] == 2:
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
    
    # For multi-class classification
    else:
        # One-vs-Rest approach
        n_classes = y_prob.shape[1]
        
        # Binarize the output
        y_test_bin = pd.get_dummies(y_test).values
        
        # Compute ROC curve and ROC area for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {i} (area = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (One-vs-Rest)')
        plt.legend(loc="lower right")
    
    return fig


def plot_precision_recall_curve(model, X_test, y_test, figsize=(10, 8)):
    """
    Plot precision-recall curve.
    
    Parameters:
    -----------
    model : estimator
        Trained model.
    X_test : array-like
        Test features.
    y_test : array-like
        Test target.
    figsize : tuple, optional
        Figure size, by default (10, 8).
        
    Returns:
    --------
    matplotlib.figure.Figure
        Precision-recall curve plot.
    """
    # Check if model can predict probabilities
    if not hasattr(model, 'predict_proba'):
        print("Model does not support predict_proba. Cannot plot precision-recall curve.")
        return None
    
    # Get probabilities
    y_prob = model.predict_proba(X_test)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # For binary classification
    if y_prob.shape[1] == 2:
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1])
        
        # Plot precision-recall curve
        plt.plot(recall, precision, lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
    
    # For multi-class classification
    else:
        # One-vs-Rest approach
        n_classes = y_prob.shape[1]
        
        # Binarize the output
        y_test_bin = pd.get_dummies(y_test).values
        
        # Compute precision-recall curve for each class
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
            plt.plot(recall, precision, lw=2, label=f'Class {i}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (One-vs-Rest)')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend(loc="lower left")
    
    return fig


def plot_feature_importance(model, feature_names, top_n=20, figsize=(12, 10)):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    model : estimator
        Trained model.
    feature_names : list
        List of feature names.
    top_n : int, optional
        Number of top features to plot, by default 20.
    figsize : tuple, optional
        Figure size, by default (12, 10).
        
    Returns:
    --------
    matplotlib.figure.Figure
        Feature importance plot.
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute. Cannot plot feature importance.")
        return None
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create dataframe of feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    # Select top N features
    feature_importance_df = feature_importance_df.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot feature importances
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    
    # Set labels
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    
    return fig


def compare_models(models, X_test, y_test, metrics_to_compare=None):
    """
    Compare multiple models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models.
    X_test : array-like
        Test features.
    y_test : array-like
        Test target.
    metrics_to_compare : list, optional
        List of metrics to compare, by default None.
        
    Returns:
    --------
    pd.DataFrame
        Dataframe of model comparison.
    """
    if metrics_to_compare is None:
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'log_loss']
    
    # Initialize results dataframe
    results = pd.DataFrame(columns=metrics_to_compare)
    
    # Evaluate each model
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        metrics = evaluate_model(model, X_test, y_test)
        
        # Add metrics to results
        results.loc[name] = [metrics.get(metric, np.nan) for metric in metrics_to_compare]
    
    # Sort by accuracy
    results = results.sort_values('accuracy', ascending=False)
    
    return results


def plot_model_comparison(comparison_df, metrics_to_plot=None, figsize=(15, 10)):
    """
    Plot model comparison.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        Dataframe of model comparison.
    metrics_to_plot : list, optional
        List of metrics to plot, by default None.
    figsize : tuple, optional
        Figure size, by default (15, 10).
        
    Returns:
    --------
    matplotlib.figure.Figure
        Model comparison plot.
    """
    if metrics_to_plot is None:
        # Exclude log_loss from default metrics to plot
        metrics_to_plot = [col for col in comparison_df.columns if col != 'log_loss']
    
    # Create figure
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=figsize)
    
    # Plot each metric
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i] if len(metrics_to_plot) > 1 else axes
        
        # Sort by metric
        sorted_df = comparison_df.sort_values(metric, ascending=False)
        
        # Plot metric
        sns.barplot(x=sorted_df.index, y=sorted_df[metric], palette='viridis', ax=ax)
        
        # Set labels
        ax.set_title(f'{metric.capitalize()}')
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.capitalize())
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    return fig
