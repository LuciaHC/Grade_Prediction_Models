import configparser
from data_processing import data_cleaning_pipeline, final_predictions_csv, get_data
from utils import get_pred_Model_1, get_pred_Model_2


config = configparser.ConfigParser()
config.read('configuration.ini')

if __name__ == '__main__':

    scaler = data_cleaning_pipeline()
    X_train_M1, X_test_M1, X_train_M2,X_test_M2, y_train, y_test = get_data()

    predictions_Model1 = get_pred_Model_1(X_train_M1,y_train,X_test_M1)
    predictions_Model2 = get_pred_Model_2(X_train_M2,y_train,X_test_M2)

    file = config['created_files']['predictions']
    final_predictions_csv(file, predictions_Model1,predictions_Model2)


