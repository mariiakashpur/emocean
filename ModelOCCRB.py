from Model import Model
from ClassifierRB import ClassifierRB
from Evaluation import Evaluation
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix



class ModelOCCRB(Model):

    def __init__(self, X_train, y_train, X_test, y_test, encoder, X_val=None, y_val=None):
        """
        :param corpus:
        """
        Model.__init__(self, X_train, y_train, X_test, y_test, X_val, y_val)
        self.clf = ClassifierRB(encoder)


    def predict_test(self):
        self.pred_test = self.clf.predict(self.X_test)
