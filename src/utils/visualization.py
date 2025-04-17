"""
Visualization utilities for IHD diagnosis project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_categorical_distribution(data, categorical_cols=None, target_col='Ischemic Heart Disease (IHD) Status', figsize=(15, 10)):
    """
    Plot distribution of categorical features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data.
    categorical_cols : list, optional
        List of categorical columns, by default None.
    target_col : str, optional
        Target column name, by default 'Ischemic Heart Disease (IHD) Status'.
    figsize : tuple, optional
        Figure size, by default (15, 10).
        
    Returns:
    --------
    matplotlib.figure.Figure
        Distribution plot.
    """
    # If categorical_cols is None, use all object and category columns
    if categorical_cols is None:
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        # Exclude target column from categorical columns
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
    
    # Create figure
    fig, axes = plt.subplots(len(categorical_cols), 1, figsize=figsize)
    
    # Plot each categorical feature
    for i, col in enumerate(categorical_cols):
        ax = axes[i] if len(categorical_cols) > 1 else axes
        
        # Create countplot
        sns.countplot(
            data=data,
            x=col,
            hue=target_col,
            palette='coolwarm',
            ax=ax
        )
        
        # Set labels
        ax.set_title(f'{col} vs {target_col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        
        # Rotate x-axis labels if needed
        if len(data[col].unique()) > 5:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add percentage annotations
        total = len(data)
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                percentage = (height / total) * 100
                ax.annotate(
                    f'{percentage:.1f}%',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=8
                )
    
    plt.tight_layout()
    
    return fig


def plot_numerical_distribution(data, numerical_cols=None, target_col='Ischemic Heart Disease (IHD) Status', figsize=(15, 15)):
    """
    Plot distribution of numerical features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data.
    numerical_cols : list, optional
        List of numerical columns, by default None.
    target_col : str, optional
        Target column name, by default 'Ischemic Heart Disease (IHD) Status'.
    figsize : tuple, optional
        Figure size, by default (15, 15).
        
    Returns:
    --------
    matplotlib.figure.Figure
        Distribution plot.
    """
    # If numerical_cols is None, use all numeric columns
    if numerical_cols is None:
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude target column from numerical columns
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
    
    # Create figure
    fig, axes = plt.subplots(len(numerical_cols), 2, figsize=figsize)
    
    # Plot each numerical feature
    for i, col in enumerate(numerical_cols):
        # Histogram
        ax1 = axes[i, 0] if len(numerical_cols) > 1 else axes[0]
        sns.histplot(
            data=data,
            x=col,
            hue=target_col,
            kde=True,
            palette='coolwarm',
            ax=ax1
        )
        ax1.set_title(f'Distribution of {col}')
        ax1.set_xlabel(col)
        ax1.set_ylabel('Count')
        
        # Boxplot
        ax2 = axes[i, 1] if len(numerical_cols) > 1 else axes[1]
        sns.boxplot(
            data=data,
            x=target_col,
            y=col,
            palette='coolwarm',
            ax=ax2
        )
        ax2.set_title(f'{col} by {target_col}')
        ax2.set_xlabel(target_col)
        ax2.set_ylabel(col)
    
    plt.tight_layout()
    
    return fig


def plot_correlation_matrix(data, figsize=(12, 10)):
    """
    Plot correlation matrix.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data.
    figsize : tuple, optional
        Figure size, by default (12, 10).
        
    Returns:
    --------
    matplotlib.figure.Figure
        Correlation matrix plot.
    """
    # Calculate correlation matrix
    corr = data.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5},
        annot=True,
        fmt=".2f",
        ax=ax
    )
    
    # Set title
    plt.title('Correlation Matrix')
    
    return fig


def plot_pca_components(X, y, n_components=2, figsize=(10, 8)):
    """
    Plot PCA components.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix.
    y : array-like
        Target variable.
    n_components : int, optional
        Number of components, by default 2.
    figsize : tuple, optional
        Figure size, by default (10, 8).
        
    Returns:
    --------
    matplotlib.figure.Figure
        PCA plot.
    """
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y,
        cmap='coolwarm',
        alpha=0.8,
        edgecolors='k',
        s=50
    )
    
    # Add legend
    legend = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend)
    
    # Set labels
    ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax.set_title('PCA of Dataset')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig


def plot_tsne(X, y, perplexity=30, figsize=(10, 8)):
    """
    Plot t-SNE visualization.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix.
    y : array-like
        Target variable.
    perplexity : int, optional
        Perplexity parameter for t-SNE, by default 30.
    figsize : tuple, optional
        Figure size, by default (10, 8).
        
    Returns:
    --------
    matplotlib.figure.Figure
        t-SNE plot.
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    scatter = ax.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=y,
        cmap='coolwarm',
        alpha=0.8,
        edgecolors='k',
        s=50
    )
    
    # Add legend
    legend = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend)
    
    # Set labels
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_title('t-SNE Visualization')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig


def plot_learning_curve(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5), figsize=(10, 8)):
    """
    Plot learning curve.
    
    Parameters:
    -----------
    model : estimator
        Model to evaluate.
    X : array-like
        Feature matrix.
    y : array-like
        Target variable.
    cv : int, optional
        Number of cross-validation folds, by default 5.
    train_sizes : array-like, optional
        Train sizes, by default np.linspace(0.1, 1.0, 5).
    figsize : tuple, optional
        Figure size, by default (10, 8).
        
    Returns:
    --------
    matplotlib.figure.Figure
        Learning curve plot.
    """
    from sklearn.model_selection import learning_curve
    
    # Calculate learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes
    )
    
    # Calculate mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot learning curve
    ax.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    ax.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
    
    # Plot standard deviation
    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color='r'
    )
    ax.fill_between(
        train_sizes,
        test_mean - test_std,
        test_mean + test_std,
        alpha=0.1,
        color='g'
    )
    
    # Set labels
    ax.set_xlabel('Training examples')
    ax.set_ylabel('Score')
    ax.set_title('Learning Curve')
    ax.legend(loc='best')
    ax.grid(True)
    
    return fig
