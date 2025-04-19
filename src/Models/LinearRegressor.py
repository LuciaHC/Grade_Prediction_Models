import numpy as np
from sklearn.metrics import r2_score
import pandas as pd


class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
    This class is programmed specifically for the reception of data that I used for my models, further modification might be needed if you reuse this code.
    """

    def __init__(self):
        self.coefficients = None
        self.loss_history = []

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation (least squares) or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): Training method ("least_squares" or "gradient_descent").
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients in-place.
        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(f"Method {method} not available for training.")

        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float64)

        if isinstance(y, pd.Series):
            y = y.values
        y = np.asarray(y, dtype=np.float64).flatten()

        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term

        if method == "least_squares":
            self.fit_multiple(X_b, y)
        elif method == "gradient_descent":
            self.fit_gradient_descent(X_b, y, learning_rate, iterations)

    def fit_multiple(self, X, y):
        """
        Fit the model using the normal equation (least squares).
        """
        self.coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
        """
        Fit the model using gradient descent.
        """
        m, n = X.shape
        self.coefficients = np.zeros(n)

        for epoch in range(iterations):
            predictions = X @ self.coefficients
            error = predictions - y
            gradient = (X.T @ error) / m
            self.coefficients -= learning_rate * gradient

            # Save loss for monitoring convergence
            mse = np.mean(error**2)
            self.loss_history.append(mse)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}: MSE = {mse}")

    def predict(self, X):
        """
        Predict values using the fitted model.
        """
        if self.coefficients is None:
            raise ValueError("Model is not yet fitted.")

        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.coefficients

    def score(self, X, y, sample_weights=None):
        """Return the coefficient of determination of the prediction.

        Args:
            X (array-like): test samples
            y (array-like): y true values
            sample_weights (arra-like, optional): Sample weights. Defaults to None.
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weights)
