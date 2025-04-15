import numpy as np
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from cvxopt import matrix, solvers


class LogisticRegressor:
    def __init__(self,learning_rate = 0.01,penalty = None):
        """
        Initializes the Logistic Regressor model.

        Args:
        - learning_rate (float): The step size at each iteration while moving toward a minimum of the
                            loss function.
        - penalty (str): Type of regularization (None, 'lasso', 'ridge', 'elasticnet'). Default is None.

        Attributes:
        - weights (np.ndarray): A placeholder for the weights of the model.
                                These will be initialized in the training phase.
        - bias (float): A placeholder for the bias of the model.
                        This will also be initialized in the training phase.
        """
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.penalty=penalty

    def fit(
        self,
        X,
        y,
        num_iterations=1000,
        l1_ratio=0.5,
        C=1.0,
        verbose=False,
        print_every=100,
    ):
        """
        Fits the logistic regression model to the data using gradient descent.

        This method initializes the model's weights and bias, then iteratively updates these parameters by
        moving in the direction of the negative gradient of the loss function (computed using the
        log_likelihood method).


        Parameters:
        - X (np.ndarray): The input features, with shape (m, n), where m is the number of examples and n is
                            the number of features.
        - y (np.ndarray): The true labels of the data, with shape (m,).
        - num_iterations (int): The number of iterations for which the optimization algorithm should run.
        - l1_ratio (float): The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
                            l1_ratio=0 corresponds to L2 penalty,
                            l1_ratio=1 to L1. Only used if penalty='elasticnet'.
                            Default is 0.5.
        - C (float): Inverse of regularization strength; must be a positive float.
                            Smaller values specify stronger regularization.
        - verbose (bool): Print loss every print_every iterations.
        - print_every (int): Period of number of iterations to show the loss.

        Updates:
        - self.weights: The weights of the model after training.
        - self.bias: The bias of the model after training.
        """

        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float64)

        if isinstance(y, pd.Series):
            y = y.values
        y = np.asarray(y, dtype=np.float64).flatten()

        m, n = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Encode the classes of the target variablee
        lb = LabelBinarizer()
        Y = lb.fit_transform(y)


        self.weights = np.zeros((n_classes, n))
        self.bias = np.zeros(n_classes)

        for i in range(num_iterations): 
            y_hat = self.predict_proba(X)
            # Compute loss
            loss = self.log_likelihood(Y, y_hat)
            error = y_hat - Y

            # Logging
            if i % print_every == 0 and verbose:
                print(f"Iteration {i}: Loss {loss}")

            dw = (1 / m) * np.dot(error.T, X)  
            db = (1 / m) * np.sum(error, axis=0)  

            if self.penalty == "lasso":
                dw = self.lasso_regularization(dw, m, C)
            elif self.penalty == "ridge":
                dw = self.ridge_regularization(dw, m, C)
            elif self.penalty == "elasticnet":
                dw = self.elasticnet_regularization(dw, m, C, l1_ratio)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        """
        Predicts probability estimates for all classes for each sample X.

        Parameters:
        - X (np.ndarray): The input features, with shape (m, n), where m is the number of samples and
            n is the number of features.

        Returns:
        - A numpy array of shape (m, 1) containing the probability of the positive class for each sample.
        """
        z = self.bias + np.dot(X, self.weights.T)
        return self.softmax(z)

    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)  # numerical stability
        exp_z = np.exp(z.astype(float))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def predict(self, X):
        """
        Predicts class labels for samples in X.

        Parameters:
        - X (np.ndarray): The input features, with shape (m, n), where m is the number of samples and n
                            is the number of features.

        Returns:
        - A numpy array of shape (m,) containing the class label (0 or 1) for each sample.
        """
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)

        return self.classes[predictions]

    def lasso_regularization(self, dw, m, C):
        """
        Applies L1 regularization (Lasso) to the gradient during the weight update step in gradient descent.
        L1 regularization encourages sparsity in the model weights, potentially setting some weights to zero,
        which can serve as a form of feature selection.

        The L1 regularization term is added directly to the gradient of the loss function with respect to
        the weights. This term is proportional to the sign of each weight, scaled by the regularization
        strength (C) and inversely proportional to the number of samples (m).

        Parameters:
        - dw (np.ndarray): The gradient of the loss function with respect to the weights, before regularization.
        - m (int): The number of samples in the dataset.
        - C (float): Inverse of regularization strength; must be a positive float.
                    Smaller values specify stronger regularization.

        Returns:
        - np.ndarray: The adjusted gradient of the loss function with respect to the weights,
                      after applying L1 regularization.
        """
        lasso_gradient = (C / m) * np.sign(self.weights)
        return np.array(dw + lasso_gradient)

    def ridge_regularization(self, dw, m, C):
        """
        Applies L2 regularization (Ridge) to the gradient during the weight update step in gradient descent.
        L2 regularization penalizes the square of the weights, which discourages large weights and helps to
        prevent overfitting by promoting smaller and more distributed weight values.

        The L2 regularization term is added to the gradient of the loss function with respect to the weights
        as a term proportional to each weight, scaled by the regularization strength (C) and inversely
        proportional to the number of samples (m).

        Parameters:
        - dw (np.ndarray): The gradient of the loss function with respect to the weights, before regularization.
        - m (int): The number of samples in the dataset.
        - C (float): Inverse of regularization strength; must be a positive float.
                     Smaller values specify stronger regularization.

        Returns:
        - np.ndarray: The adjusted gradient of the loss function with respect to the weights,
                        after applying L2 regularization.
        """
        ridge_gradient = (C / m) * self.weights
        return np.array(dw + ridge_gradient)

    def elasticnet_regularization(self, dw, m, C, l1_ratio):
        """
        Applies Elastic Net regularization to the gradient during the weight update step in gradient descent.
        Elastic Net combines L1 and L2 regularization, incorporating both the sparsity-inducing properties
        of L1 and the weight shrinkage effect of L2. This can lead to a model that is robust to various types
        of data and prevents overfitting.

        The regularization term combines the L1 and L2 terms, scaled by the regularization strength (C) and
        the mix ratio (l1_ratio) between L1 and L2 regularization. The term is inversely proportional to the
        number of samples (m).

        Parameters:
        - dw (np.ndarray): The gradient of the loss function with respect to the weights, before regularization.
        - m (int): The number of samples in the dataset.
        - C (float): Inverse of regularization strength; must be a positive float.
                     Smaller values specify stronger regularization.
        - l1_ratio (float): The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0 corresponds
                            to L2 penalty, l1_ratio=1 to L1. Only used if penalty='elasticnet'.
                            Default is 0.5.

        Returns:
        - np.ndarray: The adjusted gradient of the loss function with respect to the weights,
                      after applying Elastic Net regularization.
        """
        lasso_gradient = (C / m) * np.sign(self.weights)
        ridge_gradient = (C / m) * self.weights
        elasticnet_gradient = l1_ratio * lasso_gradient + (1 - l1_ratio) * ridge_gradient
        return np.array(dw + elasticnet_gradient)

    @staticmethod
    def log_likelihood(y, y_hat):
        """
        Computes the Log-Likelihood loss for logistic regression, which is equivalent to
        computing the cross-entropy loss between the true labels and predicted probabilities.
        This loss function is used to measure how well the model predicts the actual class
        labels. 

        Parameters:
        - y (np.ndarray): The true labels of the data. Should be a 1D array of binary values (0 or 1).
        - y_hat (np.ndarray): The predicted probabilities of the data belonging to the positive class (1).
                            Should be a 1D array with values between 0 and 1.

        Returns:
        - The computed loss value as a scalar.
        """
        m = y.shape[0]
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        log_likelihood = -np.sum(y * np.log(y_hat)) / m

        return log_likelihood

    @staticmethod
    def sigmoid(z):
        """
        Computes the sigmoid of z, a scalar or numpy array of any size. The sigmoid function is used as the
        activation function in logistic regression, mapping any real-valued number into the range (0, 1),
        which can be interpreted as a probability. It is defined as 1 / (1 + exp(-z)), where exp(-z)
        is the exponential of the negative of z.

        Parameters:
        - z (float or np.ndarray): Input value or array for which to compute the sigmoid function.

        Returns:
        - The sigmoid of z.
        """
        sigmoid_value = 1/(1 + np.exp(-z.astype(float)))
        return sigmoid_value
    
    def score(self,X,y,sample_weights=None):
        """Return the coefficient of determination of the prediction.

        Args:
            X (array-like): test samples
            y (array-like): y true values
            sample_weights (arra-like, optional): Sample weights. Defaults to None.
        """        
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weights)


class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
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
        self.coefficients = np.zeros(n)  # Initialize parameters at zero

        for epoch in range(iterations):
            predictions = X @ self.coefficients
            error = predictions - y
            gradient = (X.T @ error) / m  # Compute gradient
            self.coefficients -= learning_rate * gradient  # Update parameters

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

        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        return X_b @ self.coefficients
    
    def score(self,X,y,sample_weights=None):
        """Return the coefficient of determination of the prediction.

        Args:
            X (array-like): test samples
            y (array-like): y true values
            sample_weights (arra-like, optional): Sample weights. Defaults to None.
        """        
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weights)

class SVR_manual:
    """
    Support Vector Regression (SVR) using dual optimization and kernel methods.

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
        q = matrix(np.hstack([self.epsilon - y, self.epsilon + y]).astype(np.double))

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
        self.alpha = sol[:n] # positive slack variables
        self.alpha_star = sol[n:] # negative slack variables

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
    
    def score(self,X,y,sample_weights=None):
        """Return the score.

        Args:
            X (array-like): test samples
            y (array-like): y true values
            sample_weights (arra-like, optional): Sample weights. Defaults to None.
        """        
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weights)

