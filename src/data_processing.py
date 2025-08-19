import numpy as np
import pandas as pd
import configparser
import os
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


config = configparser.ConfigParser()
config.read('configuration.ini')


def get_data():
    """Fetches and preprocesses the data.        
    """
    numerical_vars = ['edad', 'suspensos', 'RelFam', 'TiempoLib', 'Medu', 'Pedu', 'TiempoViaje',
                      'TiempoEstudio', 'SalAm', 'AlcSem', 'AlcFin', 'salud', 'faltas', 'T1', 'T2']

    data = pd.read_csv('data/train.csv', sep=',')
    X_train = data.drop(columns=['T3']).reset_index(drop=True)
    y_train = data['T3'].reset_index(drop=True)

    data_test = pd.read_csv('data/test.csv', sep=',')
    X_test = data_test.drop(columns=['T3']).reset_index(drop=True)
    y_test = data_test['T3'].reset_index(drop=True)

    scaler_X = StandardScaler()
    X_train_M1 = X_train.copy()
    X_test_M1 = X_test.copy()
    X_train_M1[numerical_vars] = scaler_X.fit_transform(
        X_train[numerical_vars])
    X_test_M1[numerical_vars] = scaler_X.transform(X_test[numerical_vars])

    X_train_M2 = X_train_M1.drop(columns=['T2', 'T1']).reset_index(drop=True)
    X_test_M2 = X_test_M1.drop(columns=['T2', 'T1']).reset_index(drop=True)

    return X_train_M1, X_test_M1, X_train_M2, X_test_M2, y_train, y_test


def standarize_numerical_variables(X_train, X_test, y_train, y_test, model):
    """Standardizes the numerical variables in the dataset.

    Args:
        X_train (DataFrame): DataFrame containing training features.
        X_test (DataFrame): DataFrame containing testing features.
        y_train (Series): Series containing training labels.
        y_test (Series): Series containing testing labels.
        model (int): Model identifier (1 or 2).

    Returns:
        Datasets ready for training and testing.
    """
    if model == 1:
        numerical_vars = ['edad', 'suspensos', 'RelFam', 'TiempoLib', 'Medu', 'Pedu', 'TiempoViaje',
                          'TiempoEstudio', 'SalAm', 'AlcSem', 'AlcFin', 'salud', 'faltas', 'T1', 'T2']
    else:
        numerical_vars = ['edad', 'suspensos', 'RelFam', 'TiempoLib', 'Medu', 'Pedu',
                          'TiempoViaje', 'TiempoEstudio', 'SalAm', 'AlcSem', 'AlcFin', 'salud', 'faltas']

    scaler_X = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numerical_vars] = scaler_X.fit_transform(
        X_train[numerical_vars])
    X_test_scaled[numerical_vars] = scaler_X.transform(X_test[numerical_vars])

    scaler_y = StandardScaler()
    y_train_scaled = pd.Series(scaler_y.fit_transform(
        y_train.values.reshape(-1, 1)).ravel(), index=y_train.index)
    y_test_scaled = pd.Series(scaler_y.transform(
        y_test.values.reshape(-1, 1)).ravel(), index=y_test.index)

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_y


def data_cleaning_pipeline():
    """Runs the data cleaning pipeline."""
    # numerical_vars = ['edad','suspensos','RelFam','TiempoLib','Medu','Pedu','TiempoViaje','TiempoEstudio','SalAm','AlcSem','AlcFin','salud','faltas','T1','T2','T3']
    categorical_vars = ['escuela', 'sexo', 'entorno', 'TamFam', 'EstPadres', 'Mtrab', 'Ptrab', 'razon', 'tutor',
                        'apoyo', 'ApFam', 'academia', 'extras', 'enfermeria', 'EstSup', 'internet', 'pareja', 'asignatura']

    os.makedirs('data', exist_ok=True)

    file1 = config['files']['test']
    df_test = pd.read_csv(file1, sep=',')

    file2 = config['files']['train']
    df_train = pd.read_csv(file2, sep=',')

    # Outliers
    df_train['faltas'] = df_train['faltas'].clip(0, 150)
    df_test['faltas'] = df_test['faltas'].clip(0, 150)

    # Missing Values
    for column in ['TiempoEstudio', 'RelFam', 'AlcSem']:
        df_train[column] = df_train[column].fillna(
            value=df_train[column].mode()[0])
    imp = IterativeImputer(max_iter=10, random_state=0)
    df_train[["Medu", "Pedu"]] = imp.fit_transform(df_train[["Medu", "Pedu"]])

    for column in ['TiempoEstudio', 'RelFam', 'AlcSem']:
        df_test[column] = df_test[column].fillna(
            value=df_test[column].mode()[0])
    imp = IterativeImputer(max_iter=10, random_state=0)
    df_test[["Medu", "Pedu"]] = imp.fit_transform(df_test[["Medu", "Pedu"]])

    # Correct errors in data
    df_train.loc[df_train['razon'] == 'otros', 'razon'] = 'otras'
    df_test.loc[df_test['razon'] == 'otros', 'razon'] = 'otras'

    # Dummy encoding
    df_train_cat = pd.get_dummies(
        df_train, columns=categorical_vars, drop_first=True)
    df_test_cat = pd.get_dummies(
        df_test, columns=categorical_vars, drop_first=True)

    df_train_cat.to_csv("data/train.csv", sep=",", index=False)
    df_test_cat.to_csv("data/test.csv", sep=",", index=False)


def inverse_scale_T3(scaler, predictions):
    """Reverts process of standarization given a scaler

    Args:
        scaler (StandardScaler): scaler used
        predictions (array-like): array whre the unstandarization is done

    Returns:
        array-like: unstandarized array
    """
    if isinstance(predictions, np.ndarray):
        predictions = pd.Series(predictions)

    mean = scaler.mean_
    scale = scaler.scale_

    original_col = predictions * scale + mean
    return original_col


def final_predictions_csv(file, model1_predictions, model2_predictions):
    """Saves the final predictions to a CSV file.

    Args:
        file (str): The path to the output CSV file.
        model1_predictions (array-like): Predictions from the first model.
        model2_predictions (array-like): Predictions from the second model.
    """
    submission = pd.DataFrame(
        {'Modelo_i': model1_predictions, 'Modelo_ii': model2_predictions})
    submission = submission.clip(lower=0)
    submission.to_csv(file, index=False)


def get_path(section, key, base_path):
    """Manages path routes in the notebooks

    Args:
        section (str): section in configuration.ini
        key (str): key in configuration.ini
        base_path (str): base path of the notebook

    Returns:
        os.path: joined path so python can find the routes
    """
    config = configparser.ConfigParser()
    config.read(os.path.join(base_path, 'configuration.ini'))

    rel_path = config[section][key]
    return os.path.join(base_path, rel_path)
