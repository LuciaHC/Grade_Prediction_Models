import configparser
from data_processing import data_cleaning_pipeline



if __name__ == '__main__':

    # config = configparser.ConfigParser()
    # config.read('configuration.ini')
    # predictions_file = config['created_files']['predictions']

    scaler = data_cleaning_pipeline()

    # from functions import cross_validation
    # from sklearn.linear_model import LinearRegression
    # from sklearn.linear_model import LogisticRegression
    # import pandas as pd

    # data = pd.read_csv(config['created_files']['train'],sep = ',')
    # X_train = data.drop(columns=['T3']).reset_index(drop=True)
    # y_train = data['T3'].reset_index(drop=True)
    # y_train = f.inverse_scale_T3(scaler,y_train)


    # data = pd.read_csv(config['created_files']['test'],sep = ',')
    # X_test = data.drop(columns=['T3']).reset_index(drop=True)
    # y_test = data['T3'].reset_index(drop=True)

    # LogRegSK = LogisticRegression()
    # LogRegSK.fit(X_train,y_train)

    # predictions = LogRegSK.predict(X_train)
    # print(predictions)

    # print(LogRegSK.score(X_train,y_train))


    

    #print(LinRegSK.score(X_train,y_train))

    # # Load and train Model1
    # model_1 = Model1()
    # model_1.fit()
    # model1_final_pred = model_1.predict()

    # # Load and train Model2
    # model_2 = Model2()
    # model_2.fit()
    # model2_final_pred = model_2.predict()

    # # Write predictions in a final file
    # f.final_predictions_csv(predictions_file,model1_final_pred,model2_final_pred)