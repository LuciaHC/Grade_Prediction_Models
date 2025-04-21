import numpy as np
from sklearn.model_selection import KFold
from utils import evaluate_regression_metrics
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)


class StackRegressor:
    """
    Stacking regressor using 2 layers. It is trained to optimally combine the model predictions to form a new set of predictions.
    This class is programmed specifically for the reception of data that I used for my models, further modification might be needed if you reuse this code.
    """

    def __init__(self, estimators, final_estimator,cv=5):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv

    def fit(self,train,test,):

        columns = train.columns.drop('T3')

        pred = []
        for _ in range(len(self.estimators)):
            pred.append(np.zeros(train.shape[0]))

        test = []
        for _ in range(len(self.estimators)):
            test.append(np.zeros(test.shape[0]))

        kf = KFold(n_splits=self.cv,random_state=48,shuffle=True)

        # Iterate foldss
        for trn_idx, test_idx in kf.split(train[columns],train['T3']):

            X_tr,X_val=train[columns].iloc[trn_idx],train[columns].iloc[test_idx]
            y_tr,y_val=train['T3'].iloc[trn_idx],train['T3'].iloc[test_idx]

            n = 0
            for model in self.estimators:
                model.fit(X_tr,y_tr)
                pred[n][test_idx] = model.predict(X_val)
                test[n] += model.predict(test[columns])/kf.n_splits
                n += 1

        self.stacked_predictions = np.column_stack(pred)
        self.stacked_test_predictions = np.column_stack(test)


    def predict(self,train,test):
        kf = KFold(n_splits=self.cv,random_state=48,shuffle=True)
        final_prediction = np.zeros(test.shape[0])

        self.scores = []
        self.MAE = []
        self.MSE = []
        for trn_idx, test_idx in kf.split(self.stacked_predictions,train['T3']):
            X_tr,X_val=self.stacked_predictions[trn_idx],self.stacked_predictions[test_idx]
            y_tr,y_val=train['T3'].iloc[trn_idx],train['T3'].iloc[test_idx]
            
            self.final_estimator.fit(X_tr,y_tr)
            
            final_prediction += self.final_estimator.predict(self.stacked_test_predictions)/kf.n_splits
            
            self.scores.append(r2_score(y_val, self.final_estimator.predict(X_val)))
            self.MAE.append(mean_absolute_error(y_val, self.final_estimator.predict(X_val)))
            self.MSE.append(mean_squared_error(y_val, self.final_estimator.predict(X_val)))

    def score(self):
        return {
        "Mean Absolute Error:": np.mean(self.MAE),
        "Mean Squared Error:": np.mean(self.MSE),
        "R² Score:": np.mean(self.scores),
        "R² Score std:": np.std(self.scores),
    }

