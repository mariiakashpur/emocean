from sklearn import linear_model
from sklearn.base import BaseEstimator, ClassifierMixin

class ClassifierMaxEnt(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.clf = linear_model.LogisticRegression()

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)


    def predict(self, X_test):
        return self.clf.predict(X_test)
