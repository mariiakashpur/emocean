from random import shuffle
from collections import defaultdict
import copy



class Corpus(object):

    def __init__(self, instances, labels):

        """
        :param instances:
        :param labels:
        """
        self.labels = labels
        self.instances = instances
        self.shuffled_instances = None
        self.stats = {}

    def get_instances(self):
        """

        :return:
        """
        return self.instances

    def get_shuffled_instances(self):
        """

        :return:
        """
        if self.shuffled_instances is None:
            self.shuffled_instances = copy.deepcopy(self.instances)
            shuffle(self.shuffled_instances)
        return self.shuffled_instances

    def get_train_set(self):
        """

        :return:
        """
        size = len(self.get_shuffled_instances())
        return self.get_shuffled_instances()[0:int(0.60 * size + 1)]

    def get_val_set(self):
        """

        :return:
        """
        size = len(self.get_shuffled_instances())
        return self.get_shuffled_instances()[int(0.60 * size + 1):int(0.80 * size + 1)]

    def get_test_set(self):
        """

        :return:
        """
        size = len(self.get_shuffled_instances())
        return self.get_shuffled_instances()[int(0.80 * size + 1):size]

    def get_majority_label(self):
        label_count = defaultdict(int)
        for inst in self.instances:
            label_count[inst.get_label_gold()] += 1
        return max(label_count.iterkeys(), key=(lambda key: label_count[key]))

    def get_stats(self):
        return self.stats

    def build_stats(self):
        for inst in self.instances:
            gold = inst.get_label_gold()
            predicted = inst.get_label_predicted()
            if not gold in self.get_stats():
                self.stats[gold] = {"TP": 0, "FP": 0, "FN": 0}
            if not predicted in self.get_stats:
                self.stats[predicted] = {"TP": 0, "FP": 0, "FN": 0}
            if inst.is_labeled_correctly():
                self.stats[gold]["TP"] += 1  # gold and pred labels coincide
            else:
                self.stats[gold]["FN"] += 1  # increment FN counter for gold
                self.stats[predicted]["FP"] += 1  # increment FP counter for predicted
        return self.stats

    @staticmethod
    def get_postag_set(instances):
        postag_list = []
        for inst in instances:
            for token in inst.get_tokens():
                postag_list.append(token.get_tag())
        return list(set(postag_list))
