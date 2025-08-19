import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score)
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV


def cross_validation(model, X, y, nFolds):
    """
    Perform cross-validation on a given machine learning model to evaluate its performance.

    This function manually implements n-fold cross-validation if a specific number of folds is provided.
    If nFolds is set to -1, Leave One Out (LOO) cross-validation is performed instead, which uses each
    data point as a single test set while the rest of the data serves as the training set.

    Parameters:
    - model: scikit-learn-like estimator
        The machine learning model to be evaluated. This model must implement the .fit() and .score() methods
        similar to scikit-learn models.
    - X: array-like of shape (n_samples, n_features)
        The input features to be used for training and testing the model.
    - y: array-like of shape (n_samples,)
        The target values (class labels in classification, real numbers in regression) for the input samples.
    - nFolds: int
        The number of folds to use for cross-validation. If set to -1, LOO cross-validation is performed.

    Returns:
    - mean_score: float
        The mean score across all cross-validation folds.
    - std_score: float
        The standard deviation of the scores across all cross-validation folds, indicating the variability
        of the score across folds.
    """
    if nFolds == -1:
        # Implement Leave One Out CV
        nFolds = X.shape[0]
    fold_size = X.shape[0]//nFolds

    accuracy_scores = []

    for i in range(nFolds):
        valid_indices = np.arange(i*fold_size, fold_size+i*fold_size)
        train_indices = np.setdiff1d(np.arange(X.shape[0]), valid_indices)

        X_train, X_valid = X.iloc[train_indices], X.iloc[valid_indices]
        y_train, y_valid = y.iloc[train_indices], y.iloc[valid_indices]
        model.fit(X_train, y_train)

        score = model.score(X_valid, y_valid)
        accuracy_scores.append(score)
    return np.mean(accuracy_scores), np.std(accuracy_scores)


def plot_residuals_color(X, y, predictions, columns_to_plot, categorical_var=None):
    """Plots the residuals using as hue (color) the categorical variable of your choice

    Args:
        X: array-like of shape (n_samples, n_features) The input features to be used for training and testing the model.
        y: array-like of shape (n_samples,) The target values (class labels in classification, real numbers in regression) for the input samples.
        predictions: predictions of the model
        columns_to_plot: columns to plot
        categorical_var (str, optional): variable to set the color. Defaults to None.
    """

    residuals = y - predictions

    num_features = len(X.columns)
    n_rows = int(np.ceil(np.sqrt(num_features + 4)))
    n_cols = int(np.ceil((num_features + 4) / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes = axes.flatten()

    for i, col in enumerate(columns_to_plot):
        ax = axes[i]

        if categorical_var and categorical_var in X.columns:
            sns.scatterplot(x=X[col], y=residuals, ax=ax,
                            hue=X[categorical_var])
        else:
            sns.scatterplot(x=X[col], y=residuals, ax=ax)

        ax.set_title(f'Residuals vs {col}')
        ax.axhline(0, ls='--', color='r')

    # Histogram of residuals
    sns.histplot(residuals, kde=True, ax=axes[i + 1])
    axes[i + 1].set_title('Histogram of Residuals')

    # QQ-plot of residuals
    stats.probplot(residuals, dist="norm", plot=axes[i + 2])
    axes[i + 2].set_title('QQ-Plot of Residuals')

    # Hide any unused axes
    for j in range(i + 3, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def evaluate_classification_metrics(y_test, y_train, y_pred_test, y_pred_train):
    """
    Calculate various evaluation metrics for a classification model.

    Args:
        y_test (array-like): True labels of the data (test).
        y_traim (array-like): True labels of the data (train).
        y_pred_test (array-like): Predicted labels by the model with X_test.
        y_pred_train (array-like): Predicted labels by the model with X_train.

    Returns:
        dict: A dictionary containing various evaluation metrics.
    """
    return {
        "Accuracy Test": accuracy_score(y_test, y_pred_test),
        "Precision": precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
        "F1 Score": f1_score(y_test, y_pred_test, average='weighted'),
        "Accuracy Train": accuracy_score(y_train, y_pred_train),
    }


def evaluate_regression_metrics(y_test, y_train, y_pred, y_pred_train):
    """
    Calculate various evaluation metrics for a regression model.

    Args:
        y_test (array-like): True labels of the data (test).
        y_traim (array-like): True labels of the data (train).
        y_pred (array-like): Predicted labels by the model with X_test.
        y_pred_train (array-like): Predicted labels by the model with X_train.

    Returns:
        dict: A dictionary containing various evaluation metrics.
    """
    return {
        "Mean Absolute Error:": mean_absolute_error(y_test, y_pred),
        "Mean Squared Error:": mean_squared_error(y_test, y_pred),
        "R² Score Test:": r2_score(y_test, y_pred),
        "R² Score Train:": r2_score(y_train, y_pred_train),
    }


def get_pred_Model_1(X_train, y_train, X_test):
    """Get predictions from Model 1.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_test (array-like): Test features.

    Returns:
        array-like: Model predictions.
    """
    base_models = [
        ('gbr', GradientBoostingRegressor(n_estimators=50,
         max_features=0.8, subsample=0.9, random_state=42)),
        ('RF', RandomForestRegressor(random_state=42,
         n_estimators=100, max_features=0.8)),
        ('bag', BaggingRegressor(random_state=42, n_estimators=100))
    ]
    model = StackingRegressor(estimators=base_models,
                              final_estimator=LassoCV(), cv=5, passthrough=True)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions


def get_pred_Model_2(X_train, y_train, X_test):
    """Get predictions from Model 2.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_test (array-like): Test features.

    Returns:
        array-like: Model predictions.
    """
    base_models = [
        ('gbr', GradientBoostingRegressor(n_estimators=100,
         max_features=0.8, subsample=0.5, random_state=42)),
        ('RF', RandomForestRegressor(random_state=42,
         n_estimators=200, max_features=0.8)),
        ('bag', BaggingRegressor(random_state=42, n_estimators=200))]
    model = StackingRegressor(estimators=base_models,
                              final_estimator=LassoCV(), cv=5, passthrough=True)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions
