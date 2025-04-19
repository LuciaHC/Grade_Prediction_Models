from sklearn.linear_model import LinearRegression
from utils import evaluate_regression_metrics
import numpy as np



class LR_Relations:
    """
    Linear Regression model that captures relationships in the data as seen in exploratory data analysis.
    This class is programmed specifically for the reception of data that I used for my models, this model only works for this data.
    """

    def __init__(self):
        self.predictions = []
        self.predictions_train = []

    def fit(self,X_train,X_test,y_train):
        X_train_copy,X_test_copy  = X_train.copy(), X_test.copy()
        self.model0 = LinearRegression()
        self.model0.fit(X_train_copy,y_train)
        predictions0 = self.model0.predict(X_test_copy)
        predictions_t0 = self.model0.predict(X_train_copy)
        
        # New relations (sometimes some of them make the performance worse, so we will choose)
        X_train_copy['n'] = X_train['escuela_IC']*X_train['entorno_U']
        X_test_copy['n'] = X_test['escuela_IC']*X_test['entorno_U']
        self.model1 = LinearRegression()
        self.model1.fit(X_train_copy,y_train)
        predictions1 = self.model1.predict(X_test_copy)
        predictions_t1 = self.model1.predict(X_train_copy)

        X_train_copy['t'] = X_train['T1']*X_train['T2']
        X_test_copy['t'] = X_test['T1']*X_test['T2']
        self.model2 = LinearRegression()
        self.model2.fit(X_train_copy,y_train)
        predictions2 = self.model2.predict(X_test_copy)
        predictions_t2 = self.model2.predict(X_train_copy)

        X_train_copy['m'] = X_train['Medu']*X_train['Pedu']
        X_test_copy['m'] = X_test['Medu']*X_test['Pedu']
        self.model3 = LinearRegression()
        self.model3.fit(X_train_copy,y_train)
        predictions3 = self.model3.predict(X_test_copy)
        predictions_t3 = self.model3.predict(X_train_copy)

        X_train_copy['f'] = X_train['faltas']*X_train['salud']
        X_test_copy['f'] = X_test['faltas']*X_test['salud']
        self.model4 = LinearRegression()
        self.model4.fit(X_train_copy,y_train)
        predictions4 = self.model4.predict(X_test_copy)
        predictions_t4 = self.model4.predict(X_train_copy)

        X_train_copy['a'] = X_train['AlcSem']*X_train['AlcFin']
        X_test_copy['a'] = X_test['AlcSem']*X_test['AlcFin']
        self.model5 = LinearRegression()
        self.model5.fit(X_train_copy,y_train)
        predictions5 = self.model5.predict(X_test_copy)
        predictions_t5 = self.model5.predict(X_train_copy)

        self.predictions = [predictions0,predictions1,predictions2,predictions3,predictions4,predictions5]
        self.predictions_train = [predictions_t0,predictions_t1,predictions_t2,predictions_t3,predictions_t4,predictions_t5]

    def predict(self,y_test,y_train):
        max_score = 0
        index = 0
        score_m = 0
        for i in range(len(self.predictions)):
            score = evaluate_regression_metrics(y_test,y_train,self.predictions[i],self.predictions_train[i])
            if score['R² Score Test:'] > max_score:
                max_score = score['R² Score Test:']
                score_m = score
                index = i
        return self.predictions[index], score_m
        

class LR_ensemble:
    """
    Linear Regression  Ensemble model.
    This class is programmed specifically for the reception of data that I used for my models, this model only works for this data.
    """

    def __init__(self):
        self.predictions = []
        self.predictions_train = []

    def fit(self, X_train, X_test, y_train, n_estimators=25, max_samples=0.8):
        """Fit the Linear Regression Ensemble 

        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): True test target
            n_estimators (int): Number of trees. Defaults to 10
            max_samples (float): Proportion of bootstrap samples. Defaults to 0.8
        """
        n_total_samples = len(X_train)

        for _ in range(n_estimators):
            sample_indices = np.random.choice(n_total_samples, size=int(max_samples * n_total_samples), replace=True)
            X_bootstrap = X_train.iloc[sample_indices]
            y_bootstrap = y_train.iloc[sample_indices]

            # Out of the Bag samples to use as well
            test_indices = list(set(range(len(X_train))) - set(sample_indices))
            X_OOB = X_train.iloc[test_indices]
            y_OOB = y_train.iloc[test_indices]

            model1 = LinearRegression()
            model1.fit(X_bootstrap, y_bootstrap)
            y_pred = model1.predict(X_test)
            self.predictions.append(y_pred)
            y_pred_train = model1.predict(X_train)
            self.predictions_train.append(y_pred_train)

            model = LinearRegression()
            model.fit(X_OOB, y_OOB)
            y_pred = model.predict(X_test)
            self.predictions.append(y_pred)
            y_pred_train = model.predict(X_train)
            self.predictions_train.append(y_pred_train)

    def predict(self):
        """Return the predictions of the ensemble

        Returns:
            array: predictions of the ensemble
        """
        combined_predictions = np.array(self.predictions)
        majority_vote = np.mean(combined_predictions, axis=0)

        return majority_vote
    
    def score(self):
        scores = self.score_complete()
        return scores["R² Score Test:"]
    
    def score_complete(self, y_test, y_train):
        combined_predictions_train = np.array(self.predictions_train)
        majority_vote = self.predict()

        majority_vote_train = np.mean(combined_predictions_train, axis=0)
        return evaluate_regression_metrics(y_test, y_train, majority_vote, majority_vote_train)

