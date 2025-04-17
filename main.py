"""
Main script for IHD diagnosis project.
"""

import os
import argparse
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from src.data.preprocessing import preprocess_data, split_data
from src.data.feature_engineering import engineer_features
from src.models.train_model import train_model, save_models
from src.models.evaluate_model import (
    evaluate_model, compare_models, plot_confusion_matrix,
    plot_roc_curve, plot_precision_recall_curve, plot_feature_importance,
    plot_model_comparison
)
from src.utils.visualization import (
    plot_categorical_distribution, plot_numerical_distribution,
    plot_correlation_matrix, plot_pca_components, plot_tsne
)
from src.utils.helpers import (
    create_directory, save_figure, print_class_distribution,
    plot_class_distribution
)


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
    --------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='IHD Diagnosis Project')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, default='data/ASDS_Study_Data.sav',
                        help='Path to the data file')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility')
    
    # Feature engineering arguments
    parser.add_argument('--apply-feature-selection', action='store_true',
                        help='Apply feature selection')
    parser.add_argument('--apply-smote', action='store_true',
                        help='Apply SMOTE resampling')
    
    # Model arguments
    parser.add_argument('--models', type=str, nargs='+',
                        help='Models to train')
    parser.add_argument('--tune-hyperparameters', action='store_true',
                        help='Tune hyperparameters')
    parser.add_argument('--create-ensemble', action='store_true',
                        help='Create ensemble')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--save-figures', action='store_true',
                        help='Save figures')
    parser.add_argument('--save-models', action='store_true',
                        help='Save models')
    
    # Parse arguments
    args = parser.parse_args()
    
    return args


def main():
    """
    Main function.
    
    Returns:
    --------
    None
    """
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    create_directory(args.output_dir)
    
    # Load and preprocess data
    print("\n=== Loading and Preprocessing Data ===")
    data, label_encoders, scaler = preprocess_data(args.data_path)
    
    if data is None:
        print("Error loading data. Exiting.")
        return
    
    # Split data
    print("\n=== Splitting Data ===")
    X_train, X_test, y_train, y_test = split_data(
        data, test_size=args.test_size, random_state=args.random_state
    )
    
    # Print class distribution
    print("\n=== Class Distribution ===")
    print("Training Set:")
    print_class_distribution(y_train)
    print("\nTest Set:")
    print_class_distribution(y_test)
    
    # Plot class distribution
    if args.save_figures:
        fig = plot_class_distribution(y_train, title="Training Set Class Distribution")
        save_figure(fig, os.path.join(args.output_dir, 'train_class_distribution.png'))
        
        fig = plot_class_distribution(y_test, title="Test Set Class Distribution")
        save_figure(fig, os.path.join(args.output_dir, 'test_class_distribution.png'))
    
    # Apply feature engineering
    print("\n=== Feature Engineering ===")
    X_train, y_train, selected_features = engineer_features(
        pd.concat([X_train, y_train], axis=1),
        apply_feature_selection=args.apply_feature_selection,
        apply_smote_resampling=args.apply_smote
    )
    
    X_test, y_test, _ = engineer_features(
        pd.concat([X_test, y_test], axis=1),
        apply_feature_selection=args.apply_feature_selection,
        apply_smote_resampling=False
    )
    
    # Train models
    print("\n=== Training Models ===")
    trained_models, ensemble = train_model(
        X_train, y_train,
        models_to_train=args.models,
        tune_hyperparameters=args.tune_hyperparameters,
        create_ensemble=args.create_ensemble
    )
    
    # Save models
    if args.save_models:
        print("\n=== Saving Models ===")
        save_models(trained_models, ensemble)
        
        # Save label encoders and scaler
        joblib.dump(label_encoders, os.path.join('models', 'label_encoders.joblib'))
        joblib.dump(scaler, os.path.join('models', 'scaler.joblib'))
    
    # Evaluate models
    print("\n=== Evaluating Models ===")
    comparison_df = compare_models(trained_models, X_test, y_test)
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Evaluate ensemble
    if ensemble is not None:
        print("\n=== Evaluating Ensemble ===")
        ensemble_metrics = evaluate_model(ensemble, X_test, y_test)
        
        # Add ensemble to comparison
        comparison_df.loc['Voting Ensemble'] = [
            ensemble_metrics.get(metric, pd.NA) for metric in comparison_df.columns
        ]
    
    # Save comparison
    comparison_df.to_csv(os.path.join(args.output_dir, 'model_comparison.csv'))
    
    # Plot model comparison
    if args.save_figures:
        fig = plot_model_comparison(comparison_df)
        save_figure(fig, os.path.join(args.output_dir, 'model_comparison.png'))
    
    # Get best model
    best_model_name = comparison_df.index[0]
    best_model = trained_models.get(best_model_name, ensemble)
    
    print(f"\nBest Model: {best_model_name}")
    
    # Plot confusion matrix
    if args.save_figures:
        fig = plot_confusion_matrix(best_model, X_test, y_test)
        save_figure(fig, os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    # Plot ROC curve
    if args.save_figures:
        fig = plot_roc_curve(best_model, X_test, y_test)
        if fig is not None:
            save_figure(fig, os.path.join(args.output_dir, 'roc_curve.png'))
    
    # Plot precision-recall curve
    if args.save_figures:
        fig = plot_precision_recall_curve(best_model, X_test, y_test)
        if fig is not None:
            save_figure(fig, os.path.join(args.output_dir, 'precision_recall_curve.png'))
    
    # Plot feature importance
    if args.save_figures and hasattr(best_model, 'feature_importances_'):
        fig = plot_feature_importance(best_model, X_train.columns)
        save_figure(fig, os.path.join(args.output_dir, 'feature_importance.png'))
    
    print("\n=== Done ===")


if __name__ == '__main__':
    main()
