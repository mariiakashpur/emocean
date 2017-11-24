from sklearn.dummy import DummyClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class ClassifierMajority(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.clf = DummyClassifier(strategy='most_frequent',random_state=0)

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        return self.clf.predict(X_test)
