"""
Unit tests for model training and evaluation modules.
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from src.models.train_model import (
    get_base_models, get_param_grids, train_model_with_grid_search,
    train_all_models, create_voting_ensemble, train_model
)
from src.models.evaluate_model import (
    evaluate_model, compare_models
)


class TestModels(unittest.TestCase):
    """
    Test model training and evaluation functions.
    """
    
    def setUp(self):
        """
        Set up test data.
        """
        # Create synthetic data
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        
        # Convert to dataframe
        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        self.y = pd.Series(y, name='target')
    
    def test_get_base_models(self):
        """
        Test get_base_models function.
        """
        # Get base models
        models = get_base_models()
        
        # Check if models are returned
        self.assertTrue(len(models) > 0)
        self.assertTrue('Logistic Regression' in models)
        self.assertTrue('Random Forest' in models)
        self.assertTrue('Gradient Boosting' in models)
    
    def test_get_param_grids(self):
        """
        Test get_param_grids function.
        """
        # Get parameter grids
        param_grids = get_param_grids()
        
        # Check if parameter grids are returned
        self.assertTrue(len(param_grids) > 0)
        self.assertTrue('Logistic Regression' in param_grids)
        self.assertTrue('Random Forest' in param_grids)
        self.assertTrue('Gradient Boosting' in param_grids)
    
    def test_train_model_with_grid_search(self):
        """
        Test train_model_with_grid_search function.
        """
        # Get model and parameter grid
        models = get_base_models()
        param_grids = get_param_grids()
        
        # Train model with grid search
        model = models['Logistic Regression']
        param_grid = param_grids['Logistic Regression']
        
        # Simplify parameter grid for testing
        param_grid = {'C': [0.1, 1.0]}
        
        # Train model
        grid_search = train_model_with_grid_search(
            model, param_grid, self.X, self.y, cv=2
        )
        
        # Check if grid search is returned
        self.assertIsNotNone(grid_search)
        self.assertTrue(hasattr(grid_search, 'best_estimator_'))
        self.assertTrue(hasattr(grid_search, 'best_params_'))
        self.assertTrue(hasattr(grid_search, 'best_score_'))
    
    def test_train_all_models(self):
        """
        Test train_all_models function.
        """
        # Train all models
        models_to_train = ['Logistic Regression', 'Random Forest']
        trained_models = train_all_models(
            self.X, self.y, models_to_train=models_to_train, tune_hyperparameters=False
        )
        
        # Check if models are trained
        self.assertEqual(len(trained_models), len(models_to_train))
        self.assertTrue('Logistic Regression' in trained_models)
        self.assertTrue('Random Forest' in trained_models)
    
    def test_create_voting_ensemble(self):
        """
        Test create_voting_ensemble function.
        """
        # Train models
        models_to_train = ['Logistic Regression', 'Random Forest']
        trained_models = train_all_models(
            self.X, self.y, models_to_train=models_to_train, tune_hyperparameters=False
        )
        
        # Create voting ensemble
        ensemble = create_voting_ensemble(trained_models)
        
        # Check if ensemble is created
        self.assertIsNotNone(ensemble)
        self.assertEqual(len(ensemble.estimators), len(models_to_train))
    
    def test_train_model(self):
        """
        Test train_model function.
        """
        # Train models
        models_to_train = ['Logistic Regression', 'Random Forest']
        trained_models, ensemble = train_model(
            self.X, self.y, models_to_train=models_to_train,
            tune_hyperparameters=False, create_ensemble=True
        )
        
        # Check if models are trained
        self.assertEqual(len(trained_models), len(models_to_train))
        self.assertTrue('Logistic Regression' in trained_models)
        self.assertTrue('Random Forest' in trained_models)
        
        # Check if ensemble is created
        self.assertIsNotNone(ensemble)
    
    def test_evaluate_model(self):
        """
        Test evaluate_model function.
        """
        # Train model
        models = get_base_models()
        model = models['Logistic Regression']
        model.fit(self.X, self.y)
        
        # Evaluate model
        metrics = evaluate_model(model, self.X, self.y)
        
        # Check if metrics are calculated
        self.assertTrue('accuracy' in metrics)
        self.assertTrue('precision' in metrics)
        self.assertTrue('recall' in metrics)
        self.assertTrue('f1' in metrics)
        self.assertTrue('roc_auc' in metrics)
    
    def test_compare_models(self):
        """
        Test compare_models function.
        """
        # Train models
        models_to_train = ['Logistic Regression', 'Random Forest']
        trained_models = train_all_models(
            self.X, self.y, models_to_train=models_to_train, tune_hyperparameters=False
        )
        
        # Compare models
        comparison_df = compare_models(trained_models, self.X, self.y)
        
        # Check if comparison is calculated
        self.assertEqual(len(comparison_df), len(models_to_train))
        self.assertTrue('accuracy' in comparison_df.columns)
        self.assertTrue('precision' in comparison_df.columns)
        self.assertTrue('recall' in comparison_df.columns)
        self.assertTrue('f1' in comparison_df.columns)
        self.assertTrue('roc_auc' in comparison_df.columns)


if __name__ == '__main__':
    unittest.main()
