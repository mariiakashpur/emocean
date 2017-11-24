from sklearn import svm
from sklearn.base import BaseEstimator, ClassifierMixin

class ClassifierSVM(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.clf = svm.SVC(kernel='linear', class_weight="balanced")

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)


    def predict(self, X_test):
        return self.clf.predict(X_test)
