from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, ClassifierMixin

class ClassifierRF(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.clf = RandomForestRegressor()

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)


    def predict(self, X_test):
        return self.clf.predict(X_test)
