from Model import Model
from ClassifierSVM import ClassifierSVM
from ClassifierMaxEnt import ClassifierMaxEnt


class ModelOCC(Model):

    def __init__(self, X_train, y_train, X_test, y_test, X_val=None, y_val=None,):
        """
        :param corpus:
        """
        Model.__init__(self, X_train, y_train, X_test, y_test, X_val, y_val)
        self.clf = ClassifierSVM()
        # self.clf = ClassifierMaxEnt()

    def train(self):
        self.clf.fit(self.X_train, self.y_train)




