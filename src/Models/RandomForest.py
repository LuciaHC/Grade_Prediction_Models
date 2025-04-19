import numpy as np
from scipy.stats import mode
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from utils import evaluate_classification_metrics, evaluate_regression_metrics


class Random_Forest_ensemble:
    """
    Manually implements a Random Forest ensemble using decision trees for classification or regression.
    This class is programmed specifically for the reception of data that I used for my models, further modification might be needed if you reuse this code.
    """

    def __init__(self, tipo='reg'):
        """
        Initializes the Random Forest Ensemble using Decission Tree Regressors or Clasifiers
        Args:
            tipo (str, optional): 'clas' for classification or 'reg' for regression
        """
        self.predictions = []
        self.predictions_train = []
        self.tipo = tipo

    def fit(self, X_train, X_test, y_train, n_estimators=10, max_samples=0.8, max_features=0.8):
        """Fit the Bagging Ensemble using regression or classification

        Args:
           X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): True test target
            n_estimators (int): Number of trees. Defaults to 10
            max_samples (float): Proportion of bootstrap samples. Defaults to 0.8
            max_features (float): Proportion of features to use for each tree. Defaults to 0.8
        """
        n_total_samples = len(X_train)
        n_total_features = X_train.shape[1]
        max_features_int = int(max_features * n_total_features)

        for _ in range(n_estimators):
            sample_indices = np.random.choice(n_total_samples, size=int(
                max_samples * n_total_samples), replace=True)
            X_bootstrap = X_train.iloc[sample_indices]
            y_bootstrap = y_train.iloc[sample_indices]

            if self.tipo == 'clas':
                model = DecisionTreeClassifier(max_features=max_features_int)
            elif self.tipo == 'reg':
                model = DecisionTreeRegressor(max_features=max_features_int)
            else:
                raise ValueError("Parameter 'tipo' must be 'clas' or 'reg'.")

            model.fit(X_bootstrap, y_bootstrap)
            y_pred = model.predict(X_test)
            self.predictions.append(y_pred)
            y_pred_train = model.predict(X_train)
            self.predictions_train.append(y_pred_train)

    def predict(self):
        """Return the predictions of the RF ensemble, using different strategies for regression and classification

        Returns:
            array: predictions of the RF ensemble
        """
        combined_predictions = np.array(self.predictions)
        if self.tipo == 'clas':
            majority_vote, _ = mode(
                combined_predictions, axis=0, keepdims=False)
        else:
            majority_vote = np.mean(combined_predictions, axis=0)
        return majority_vote

    def score(self, y_test, y_train):
        combined_predictions_train = np.array(self.predictions_train)
        majority_vote = self.predict()

        if self.tipo == 'clas':
            majority_vote_train, _ = mode(
                combined_predictions_train, axis=0, keepdims=False)
            return evaluate_classification_metrics(y_test, y_train, majority_vote, majority_vote_train)
        else:
            majority_vote_train = np.mean(combined_predictions_train, axis=0)
            return evaluate_regression_metrics(y_test, y_train, majority_vote, majority_vote_train)


# def manual_Random_Forest(X_train, y_train, X_test, y_test, n_estimators=10, max_samples=0.8, max_features=0.8, tipo='clas'):
#     """
#     Manually implements a Random Forest ensemble using decision trees for classification or regression.

#     Parameters:
#         X_train (pd.DataFrame): Training features
#         y_train (pd.Series): Training target
#         X_test (pd.DataFrame): Test features
#         y_test (pd.Series): True test target
#         n_estimators (int): Number of trees
#         max_samples (float): Proportion of bootstrap samples
#         max_features (float): Proportion of features to use for each tree
#         tipo (str): 'clas' for classification or 'reg' for regression

#     Returns:
#         final_predictions (np.ndarray): Aggregated predictions on test set
#         score (dict): Evaluation metrics (custom function defined separately)
#     """
#     predictions = []
#     predictions_train = []
#     n_total_samples = len(X_train)
#     n_total_features = X_train.shape[1]
#     max_features_int = int(max_features * n_total_features)

#     for _ in range(n_estimators):
#         sample_indices = np.random.choice(n_total_samples, size=int(max_samples * n_total_samples), replace=True)
#         X_bootstrap = X_train.iloc[sample_indices]
#         y_bootstrap = y_train.iloc[sample_indices]

#         if tipo == 'clas':
#             model = DecisionTreeClassifier(max_features=max_features_int)
#         elif tipo == 'reg':
#             model = DecisionTreeRegressor(max_features=max_features_int)
#         else:
#             raise ValueError("Parameter 'tipo' must be 'clas' or 'reg'.")

#         model.fit(X_bootstrap, y_bootstrap)
#         y_pred = model.predict(X_test)
#         predictions.append(y_pred)
#         y_pred_train = model.predict(X_train)
#         predictions_train.append(y_pred_train)

#     combined_predictions = np.array(predictions)
#     combined_predictions_train = np.array(predictions_train)
#     if tipo == 'clas':
#         majority_vote, _ = mode(combined_predictions, axis=0, keepdims=False)
#         majority_vote_train, _ = mode(combined_predictions_train, axis=0, keepdims=False)
#         score = evaluate_classification_metrics(y_test,y_train,majority_vote,majority_vote_train)
#     else:
#         majority_vote = np.mean(combined_predictions,axis = 0)
#         majority_vote_train = np.mean(combined_predictions_train,axis = 0)
#         score = evaluate_regression_metrics(y_test,y_train,majority_vote, majority_vote_train)

#     return majority_vote, score
