import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelBinarizer


class LogisticRegressor:
    """ 
    Multiclass Logistic Regressor.
    This class is programmed specifically for the reception of data that I used for my models, further modification might be needed if you reuse this code.
    """

    def __init__(self, learning_rate=0.01, penalty=None):
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
        self.penalty = penalty

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
        elasticnet_gradient = l1_ratio * lasso_gradient + \
            (1 - l1_ratio) * ridge_gradient
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

    def score(self, X, y, sample_weights=None):
        """Return the coefficient of determination of the prediction.

        Args:
            X (array-like): test samples
            y (array-like): y true values
            sample_weights (arra-like, optional): Sample weights. Defaults to None.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weights)
