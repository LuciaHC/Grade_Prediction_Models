import numpy as np
from sklearn.metrics import r2_score
import pandas as pd
from cvxopt import matrix, solvers


class SVR_manual:
    """
    Support Vector Regression (SVR) using dual optimization and kernel methods.
    This class is programmed specifically for the reception of data that I used for my models, further modification might be needed if you reuse this code.

    Observation: I only did the regressor since the sklearn classificator performed poorly.
    """

    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1, gamma=0.1):
        """
        Initialize the SVR model.

        Parameters:
        - kernel (str): Kernel type ('linear', 'poly', 'rbf')
        - C (float): Regularization parameter
        - epsilon (float): ε-insensitive loss
        - gamma (float): Parameter for RBF/poly kernel
        """
        self.kernel_name = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = None
        self.alpha_star = None
        self.X = None
        self.y = None
        self.b = 0

    def _kernel(self, x1, x2):
        if self.kernel_name == 'linear':
            return np.dot(x1, x2)
        elif self.kernel_name == 'poly':
            return (1 + np.dot(x1, x2)) ** self.gamma
        elif self.kernel_name == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        else:
            raise ValueError("Unknown kernel")

    def _compute_kernel_matrix(self, X):
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self._kernel(X[i], X[j])
        return K

    def fit(self, X, y):
        """
        Fit the SVR model to training data.

        Parameters:
        - X: Training features, shape (n_samples, n_features)
        - y: Target values, shape (n_samples,)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float64)

        if isinstance(y, pd.Series):
            y = y.values
        y = np.asarray(y, dtype=np.float64).flatten()

        n = X.shape[0]
        self.X = X
        self.y = y

        K = self._compute_kernel_matrix(X)

        # Solving the dual problem:
        P = np.vstack([
            np.hstack([K, -K]),
            np.hstack([-K, K])
        ])

        P = matrix(P.astype(np.double))
        q = matrix(
            np.hstack([self.epsilon - y, self.epsilon + y]).astype(np.double))

        G_std = np.vstack([
            np.eye(2 * n),
            -np.eye(2 * n)
        ])

        h_std = np.hstack([
            np.ones(2 * n) * self.C,
            np.zeros(2 * n)
        ])

        A = matrix(np.hstack([np.ones(n), -np.ones(n)]).reshape(1, -1))
        b = matrix(0.0)

        G = matrix(G_std)
        h = matrix(h_std.astype(np.double))

        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)

        sol = np.array(solution['x']).flatten()
        self.alpha = sol[:n]  # positive slack variables
        self.alpha_star = sol[n:]  # negative slack variables

        # Calculate b using KKT conditions
        self.b = 0
        for i in range(n):
            # We choose a suitable vector to extract b from the formula
            if 0 < self.alpha[i] < self.C or 0 < self.alpha_star[i] < self.C:
                self.b = y[i] - self.predict(X[i].reshape(1, -1))[0]
                break

    def predict(self, X):
        """
        Predict the output for input data X.

        Parameters:
        - X: Input features, shape (n_samples, n_features)

        Returns:
        - Predicted values, shape (n_samples,)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float64)

        y_pred = []
        for x in X:
            s = 0
            for i in range(len(self.X)):
                k = self._kernel(self.X[i], x)
                s += (self.alpha[i] - self.alpha_star[i]) * k
            y_pred.append(s + self.b)
        return np.array(y_pred)

    def score(self, X, y, sample_weights=None):
        """Return the score.

        Args:
            X (array-like): test samples
            y (array-like): y true values
            sample_weights (arra-like, optional): Sample weights. Defaults to None.
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weights)
