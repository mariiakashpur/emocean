from Model import Model
from ClassifierMajority import ClassifierMajority
from sklearn.model_selection import cross_val_predict

class ModelMajority(Model):

    def __init__(self, X_train, y_train, X_test, y_test, X_val=None, y_val=None,):
        """
        :param corpus:
        """
        Model.__init__(self, X_train, y_train, X_test, y_test, X_val, y_val)
        self.clf = ClassifierMajority()

    def train(self):
        self.clf.fit(self.X_train, self.y_train)
