import numpy as np
import pandas as pd
import configparser
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
 
config = configparser.ConfigParser()
config.read('configuration.ini')



def final_predictions_csv(file,model1_predictions, model2_predictions):
    submission = pd.DataFrame({'Modelo_i': model1_predictions,'Modelo_ii': model2_predictions})
    submission.to_csv(file, index=False)

# def standarize_numerical_variables(df_train_cat, df_test_cat,numerical_vars):
#     scaler = StandardScaler()

#     # Scale only numerical columns
#     X_train_scaled = pd.DataFrame(scaler.fit_transform(df_train_cat[numerical_vars]), columns=numerical_vars, index=df_train_cat.index)
#     X_test_scaled = pd.DataFrame(scaler.transform(df_test_cat[numerical_vars]), columns=numerical_vars, index=df_test_cat.index)

#     # Concatenate scaled numerical features with original categorical features
#     X_train_final = pd.concat([df_train_cat.drop(numerical_vars, axis=1), X_train_scaled], axis=1)
#     X_test_final = pd.concat([df_test_cat.drop(numerical_vars, axis=1), X_test_scaled], axis=1)

#     return X_train_final, X_test_final, scaler

def standarize_numerical_variables(X_train, X_test, y_train, y_test):
    numerical_vars = ['edad','suspensos','RelFam','TiempoLib','Medu','Pedu','TiempoViaje','TiempoEstudio','SalAm','AlcSem','AlcFin','salud','faltas','T1','T2']

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

def data_cleaning_pipeline(rescale_data=True):
    numerical_vars = ['edad','suspensos','RelFam','TiempoLib','Medu','Pedu','TiempoViaje','TiempoEstudio','SalAm','AlcSem','AlcFin','salud','faltas','T1','T2','T3']
    categorical_vars = ['escuela','sexo','entorno','TamFam','EstPadres','Mtrab','Ptrab','razon', 'tutor','apoyo','ApFam', 'academia', 'extras', 'enfermeria', 'EstSup', 'internet', 'pareja','asignatura']

    os.makedirs('data', exist_ok=True)
    
    file1 = config['files']['test']
    df_test = pd.read_csv(file1, sep=',')

    file2 = config['files']['train']
    df_train = pd.read_csv(file2, sep=',')

    df_train['faltas'] = df_train['faltas'].clip(0,150)
    df_train = df_train.dropna(subset = ['TiempoEstudio', 'RelFam','AlcSem'])
    for column in ['Medu', 'Pedu']:
        df_train[column] = df_train[column].fillna(value=df_train[column].mode()[0])

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
    mean = scaler.mean_[11]
    scale = scaler.scale_[11]

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
