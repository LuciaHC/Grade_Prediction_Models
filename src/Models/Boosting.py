import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

class Boosting_Regressor:
    """
    Implements a Gradient Boosting Regressor using decision trees.
    This class is programmed specifically for the reception of data that I used for my models, further modification might be needed if you reuse this code.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.init_prediction = None

    def fit(self, X, y):
        """
        Fit the gradient boosting regressor.

        Args:
            X (pd.DataFrame or np.ndarray): Feature matrix.
            y (pd.Series or np.ndarray): Target vector.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float64)

        if isinstance(y, pd.Series):
            y = y.values
        y = np.asarray(y, dtype=np.float64).flatten()

        self.models = []
        self.init_prediction = np.mean(y)  
        y_pred = np.full(y.shape, self.init_prediction)

        for _ in range(self.n_estimators):
            residual = y - y_pred
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residual)

            update = tree.predict(X)
            y_pred += self.learning_rate * update
            self.models.append(tree)

    def predict(self, X):
        """
        Predict using the trained model.

        Args:
            X (pd.DataFrame or np.ndarray): Feature matrix.
        Returns:
            np.ndarray: Predicted values.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float64)

        y_pred = np.full(X.shape[0], self.init_prediction)
        for tree in self.models:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred

    def score(self, X, y_true):
        """
        Compute R² score on given data.

        Args:
            X (pd.DataFrame or np.ndarray): Feature matrix.
            y_true (pd.Series or np.ndarray): True target values.

        Returns:
            float: R² score.
        """
        y_pred = self.predict(X)

        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        y_true = np.asarray(y_true, dtype=np.float64).flatten()

        return r2_score(y_true, y_pred)
