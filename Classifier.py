import abc

class Classifier(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def predict(self, X_test):
        pass

    def get_param_grid(self):
        pass
