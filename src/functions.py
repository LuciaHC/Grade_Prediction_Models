import numpy as np
import pandas as pd
import configparser
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import mode
from sklearn.tree import DecisionTreeRegressor, export_graphviz, DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
 
config = configparser.ConfigParser()
config.read('configuration.ini')


def final_predictions_csv(file,model1_predictions, model2_predictions):
    submission = pd.DataFrame({'Modelo_i': model1_predictions,'Modelo_ii': model2_predictions})
    submission.to_csv(file, index=False)

def standarize_numerical_variables(X_train, X_test, y_train, y_test,model):
    if model == 1:
        numerical_vars = ['edad','suspensos','RelFam','TiempoLib','Medu','Pedu','TiempoViaje','TiempoEstudio','SalAm','AlcSem','AlcFin','salud','faltas','T1','T2']
    else:
        numerical_vars = ['edad','suspensos','RelFam','TiempoLib','Medu','Pedu','TiempoViaje','TiempoEstudio','SalAm','AlcSem','AlcFin','salud','faltas']

    scaler_X = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numerical_vars] = scaler_X.fit_transform(X_train[numerical_vars])
    X_test_scaled[numerical_vars] = scaler_X.transform(X_test[numerical_vars])

    scaler_y = StandardScaler()
    y_train_scaled = pd.Series(scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel(), index=y_train.index)
    y_test_scaled = pd.Series(scaler_y.transform(y_test.values.reshape(-1, 1)).ravel(), index=y_test.index)


    # # Scale only numerical columns
    # X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[numerical_vars]), columns=numerical_vars, index=X_train.index)
    # X_test_scaled = pd.DataFrame(scaler.transform(X_test[numerical_vars]), columns=numerical_vars, index=X_test.index)
    # y_train_scaled = pd.DataFrame(scaler.transform(y_train[numerical_vars]), columns=numerical_vars, index=y_train.index)
    # y_test_scaled = pd.DataFrame(scaler.transform(y_test[numerical_vars]), columns=numerical_vars, index=y_test.index)

    # # Concatenate scaled numerical features with original categorical features
    # X_train_final = pd.concat([X_train.drop(numerical_vars, axis=1), X_train_scaled], axis=1)
    # X_test_final = pd.concat([X_test.drop(numerical_vars, axis=1), X_test_scaled], axis=1)
    # y_train_final = pd.concat([X_train.drop(numerical_vars, axis=1), y_train_scaled], axis=1)
    # y_test_final = pd.concat([X_test.drop(numerical_vars, axis=1), y_test_scaled], axis=1)

    return X_train_scaled, X_test_scaled, y_train_scaled,y_test_scaled,scaler_y

def data_cleaning_pipeline():
    numerical_vars = ['edad','suspensos','RelFam','TiempoLib','Medu','Pedu','TiempoViaje','TiempoEstudio','SalAm','AlcSem','AlcFin','salud','faltas','T1','T2','T3']
    categorical_vars = ['escuela','sexo','entorno','TamFam','EstPadres','Mtrab','Ptrab','razon', 'tutor','apoyo','ApFam', 'academia', 'extras', 'enfermeria', 'EstSup', 'internet', 'pareja','asignatura']

    os.makedirs('data', exist_ok=True)
    
    file1 = config['files']['test']
    df_test = pd.read_csv(file1, sep=',')

    file2 = config['files']['train']
    df_train = pd.read_csv(file2, sep=',')

    df_train['faltas'] = df_train['faltas'].clip(0,150)
    df_train = df_train.dropna(subset = ['TiempoEstudio', 'RelFam','AlcSem'])
    # for column in ['Medu', 'Pedu']:
    #     df_train[column] = df_train[column].fillna(value=df_train[column].mode()[0])


    imp = IterativeImputer(max_iter=10, random_state=0)
    df_train[["Medu", "Pedu"]] = imp.fit_transform(df_train[["Medu", "Pedu"]])

    df_train_cat = pd.get_dummies(df_train,columns=categorical_vars,drop_first=True)
    df_test_cat = pd.get_dummies(df_test,columns=categorical_vars,drop_first=True)

    df_train_cat.to_csv("data/train.csv", sep = ",", index = False)
    df_test_cat.to_csv("data/test.csv", sep = ",", index = False)
    
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
        valid_indices = np.arange(i*fold_size,fold_size+i*fold_size)
        train_indices = np.setdiff1d(np.arange(X.shape[0]), valid_indices)

        X_train, X_valid = X.iloc[train_indices], X.iloc[valid_indices]
        y_train, y_valid = y.iloc[train_indices], y.iloc[valid_indices]
        model.fit(X_train,y_train)

        score = model.score(X_valid, y_valid)
        accuracy_scores.append(score)
    return np.mean(accuracy_scores), np.std(accuracy_scores)

def evaluate_classification_metrics(y_true, y_pred, positive_label):
    """
    Calculate various evaluation metrics for a classification model.

    Args:
        y_true (array-like): True labels of the data.
        positive_label: The label considered as the positive class.
        y_pred (array-like): Predicted labels by the model.

    Returns:
        dict: A dictionary containing various evaluation metrics.

    Metrics Calculated:
        - Confusion Matrix: [TN, FP, FN, TP]
        - Accuracy: (TP + TN) / (TP + TN + FP + FN)
        - Precision: TP / (TP + FP)
        - Recall (Sensitivity): TP / (TP + FN)
        - Specificity: TN / (TN + FP)
        - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    """
    # Map string labels to 0 or 1
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    y_pred_mapped = np.array([1 if label == positive_label else 0 for label in y_pred])

    # Confusion Matrix
    tp = sum((y_pred_mapped == 1) & (y_true_mapped == 1))
    tn = sum((y_pred_mapped == 0) & (y_true_mapped == 0))
    fp = sum((y_pred_mapped == 1) & (y_true_mapped == 0))
    fn = sum((y_pred_mapped == 0) & (y_true_mapped == 1))

    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Precision
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0

    # Recall (Sensitivity)
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return {
        'Confusion Matrix': [tn, fp, fn, tp],
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1 Score': f1
    }

def inverse_scale_T3(scaler,predictions):
    if isinstance(predictions, np.ndarray):
        predictions = pd.Series(predictions)

    mean = scaler.mean_
    scale = scaler.scale_

    original_col = predictions * scale + mean
    return original_col

def plot_residuals_color(X, y, predictions,columns_to_plot,categorical_var=None):

    residuals = y - predictions

    num_features = len(X.columns)
    # Number of rows and columns for the subplot
    n_rows = int(np.ceil(np.sqrt(num_features + 4)))  
    n_cols = int(np.ceil((num_features + 4) / n_rows))

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # TODO: Plot each variable against the residuals
    for i, col in enumerate(columns_to_plot):
        ax = axes[i]
        
        if categorical_var and categorical_var in X.columns:
            sns.scatterplot(x = X[col], y=residuals, ax=ax, hue = X[categorical_var])
        else:
            sns.scatterplot(x = X[col], y= residuals, ax=ax)
        
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
        "Accuracy Train": accuracy_score(y_train,y_pred_train),
    }

def evaluate_regression_metrics(y_test,y_train, y_pred, y_pred_train):
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

def manual_bagging(X_train, y_train, X_test, y_test, n_estimators, max_samples, tipo):
    """
    Manually implements a bagging ensemble using decision trees for classification or regression.

    Parameters:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): True test target
        n_estimators (int): Number of trees
        max_samples (float): Proportion of bootstrap samples
        tipo (str): 'clas' for classification or 'reg' for regression

    Returns:
        final_predictions (np.ndarray): Aggregated predictions on test set
        score (dict): Evaluation metrics (custom function defined separately)
    """
    predictions = []
    predictions_train = []
    for _ in range(n_estimators):

        # Create a bootstrap sample
        n_samples = int(max_samples * len(X_train))
        sample_indices = np.random.choice(len(X_train), size=n_samples, replace=True)
        X_bootstrap = X_train.iloc[sample_indices]
        y_bootstrap = y_train.iloc[sample_indices]

        if tipo == 'clas':
            model = DecisionTreeClassifier()
        elif tipo == 'reg':
            model = DecisionTreeRegressor()
        else:
            raise ValueError("Parameter 'tipo' must be 'clas' or 'reg'.")
            
        model.fit(X_bootstrap, y_bootstrap)
        y_pred = model.predict(X_test) 
        predictions.append(y_pred)
        y_pred_train = model.predict(X_train) 
        predictions_train.append(y_pred_train)

    combined_predictions = np.array(predictions)
    combined_predictions_train = np.array(predictions_train)
    if tipo == 'clas':
        majority_vote, _ = mode(combined_predictions, axis=0, keepdims=False)
        majority_vote_train, _ = mode(combined_predictions_train, axis=0, keepdims=False)
        score = evaluate_classification_metrics(y_test,y_train, majority_vote, majority_vote_train)
    else:
        majority_vote = np.mean(combined_predictions,axis = 0) 
        majority_vote_train = np.mean(combined_predictions_train, axis=0)
        score = evaluate_regression_metrics(y_test,y_train, majority_vote, majority_vote_train)

    return majority_vote, score

def manual_Random_Forest(X_train, y_train, X_test, y_test, n_estimators=10, max_samples=0.8, max_features=0.8, tipo='clas'):
    """
    Manually implements a Random Forest ensemble using decision trees for classification or regression.

    Parameters:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): True test target
        n_estimators (int): Number of trees
        max_samples (float): Proportion of bootstrap samples
        max_features (float): Proportion of features to use for each tree
        tipo (str): 'clas' for classification or 'reg' for regression

    Returns:
        final_predictions (np.ndarray): Aggregated predictions on test set
        score (dict): Evaluation metrics (custom function defined separately)
    """
    predictions = []
    predictions_train = []
    n_total_samples = len(X_train)
    n_total_features = X_train.shape[1]
    max_features_int = int(max_features * n_total_features)

    for _ in range(n_estimators):
        sample_indices = np.random.choice(n_total_samples, size=int(max_samples * n_total_samples), replace=True)
        X_bootstrap = X_train.iloc[sample_indices]
        y_bootstrap = y_train.iloc[sample_indices]

        if tipo == 'clas':
            model = DecisionTreeClassifier(max_features=max_features_int)
        elif tipo == 'reg':
            model = DecisionTreeRegressor(max_features=max_features_int)
        else:
            raise ValueError("Parameter 'tipo' must be 'clas' or 'reg'.")

        model.fit(X_bootstrap, y_bootstrap)
        y_pred = model.predict(X_test)
        predictions.append(y_pred)
        y_pred_train = model.predict(X_train)
        predictions_train.append(y_pred_train)

    combined_predictions = np.array(predictions)
    combined_predictions_train = np.array(predictions_train)
    if tipo == 'clas':
        majority_vote, _ = mode(combined_predictions, axis=0, keepdims=False)
        majority_vote_train, _ = mode(combined_predictions_train, axis=0, keepdims=False)
        score = evaluate_classification_metrics(y_test,y_train,majority_vote,majority_vote_train)
    else:
        majority_vote = np.mean(combined_predictions,axis = 0) 
        majority_vote_train = np.mean(combined_predictions_train,axis = 0) 
        score = evaluate_regression_metrics(y_test,y_train,majority_vote, majority_vote_train)

    return majority_vote, score