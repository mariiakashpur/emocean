from abc import ABCMeta, abstractmethod
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn import svm

class Model(object):
    __metaclass__ = ABCMeta

    def __init__(self, X_train, y_train, X_test, y_test,  X_val=None, y_val=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.pred_cv = None
        self.pred_test = None

    def predict_cv(self):
        self.pred_cv = cross_val_predict(self.clf, self.X_train, self.y_train, cv=10)

    def predict_test(self):
        # self.pred_test = self.clf.predict(self.X_test)
        self.pred_test = self.get_best_estimator_svm().predict(self.X_test)

    def evaluate(self, y, pred):
        """
        Use sklearn for model evaluation
        :param y: gold labels
        :param pred: predicted labels
        """
        print "accuracy: ", accuracy_score(y, pred)
        print "recall: ", recall_score(y, pred, average='weighted')
        print "precision: ", precision_score(y, pred, average='weighted')
        print "f1_score macro: ", f1_score(y, pred, average='macro')
        print "f1_score micro: ", f1_score(y, pred, average='micro')
        print "\n classification report: \n", classification_report(y, pred)
        print "\n confusion matrix:\n", confusion_matrix(y, pred)

    def evaluate_cv(self):
        self.evaluate(self.y_train, self.pred_cv)

    def evaluate_test(self):
        self.evaluate(self.y_test, self.pred_test)

    def create_result_file_alm(self, X_test, predictions, y_test, result_file):
        mapper = {
            0:2,
            1:3,
            2:4,
            3:6,
            4:7
        }
        with open(result_file, 'w') as f:
            tolist = y_test.tolist()
            for idx, val in enumerate(tolist):
                f.write("GOLD: " + str(mapper[val]) + " PREDICTED: " + str(mapper[predictions[idx]]) +
                        " SENT: " + str(X_test.iloc[idx]['text']) + "\n")


    def create_result_file(self, X_test, predictions, y_test, result_file):
        with open(result_file, 'w') as f:
            tolist = y_test.tolist()
            for idx, val in enumerate(tolist):
                f.write("GOLD: " + str(val) + " PREDICTED: " + str(predictions[idx]) +
                        " SENT: " + str(X_test.iloc[idx]['text']) + "\n")

    def get_best_estimator_svm(self):
        param_grid = [{'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
                      {'kernel': ['poly'], 'C': [0.1, 1, 10, 100]},
                      {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10, 100]}]
        grid_search = GridSearchCV(svm.SVC(), param_grid, cv = 5)
        grid_search.fit(self.X_train, self.y_train)
        return grid_search.best_estimator_

