from __future__ import division
from collections import Counter, defaultdict, OrderedDict
import math
from Token import Token
from Instance import Instance
from Corpus import Corpus


class Evaluation(object):
    def __init__(self, predictions, gold_labels):
        self.predictions = predictions
        self.stats = self.build_stats(self.predictions, gold_labels)
        self.accuracy = 0
        self.macro = 0
        self.micro = 0
        self.eval = self.evaluate() # ensure that all needed variables are set
        #  self.wrongly_labeled = []

    def build_stats(self, predictions, gold_labels):
        self.stats = {}
        zipped = zip(predictions, gold_labels)
        for z in zipped:
            predicted = z[0]
            gold = z[1]
            if not gold in self.stats:
                self.stats[gold] = {"TP": 0, "FP": 0, "FN": 0}
            if not predicted in self.stats:
                self.stats[predicted] = {"TP": 0, "FP": 0, "FN": 0}
            if predicted == gold:
                self.stats[gold]["TP"] += 1  # gold and pred labels coincide
            else:
                self.stats[gold]["FN"] += 1  # increment FN counter for gold
                self.stats[predicted]["FP"] += 1  # increment FP counter for predicted
        return self.stats

    def count_precision(self, label):
        try:
            precision = self.stats[label]["TP"] / (self.stats[label]["TP"] + self.stats[label]["FP"])
        except ZeroDivisionError:
            precision = 0
        return precision

    def count_recall(self,label):
        try:
            recall = self.stats[label]["TP"] / (self.stats[label]["TP"] + self.stats[label]["FN"])
        except ZeroDivisionError:
            recall = 0
        return recall

    def count_fscore(self, precision, recall):
        try:
            fscore = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            fscore = 0
        return fscore

    def evaluate(self):
        results = {}
        totalTP = 0
        totalFP = 0
        totalFN = 0
        totalFscore = 0

        for label in self.stats:
            totalTP += self.stats[label]["TP"]
            totalFP += self.stats[label]["FP"]
            totalFN += self.stats[label]["FN"]

            precision = self.count_precision(label)
            recall = self.count_recall(label)
            fscore = self.count_fscore(precision, recall)

            results[label] = [precision] # precision under index 0
            results[label].append(recall) # recall under index 1
            results[label].append(fscore) # fscore under index 2
            totalFscore += fscore

        self.accuracy = totalTP / len(self.predictions)
        self.macro = totalFscore / len(results)
        self.micro = self.count_fscore(totalTP / (totalTP + totalFP), totalTP / (totalTP + totalFN))

        return results

    def format(self):
        output = "accuracy: %s\nmacroaverage: %s\nmicroaverage: %s\n" % (str(round(self.accuracy, 3)), str(round(self.macro, 3)), str(round(self.micro, 3)))
        for label in self.eval:
            output += "%s ==> precision: %s\trecall: %s\tf-score: %s\n" % (label, str(round(self.eval[label][0], 3)).ljust(5),
                                                str(round(self.eval[label][1], 3)).ljust(5), str(round(self.eval[label][2], 3)).ljust(5))
        print output






