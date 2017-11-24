import os
import sys
from itertools import izip
from nltk import pos_tag, word_tokenize
from nltk.parse.stanford import StanfordDependencyParser


class Instance(object):

    def __init__(self, text, label_gold, deps=None, label_predicted=None):
        """

        :param text:
        :param label_gold:
        :param label_predicted:
        """
        self.text = text
        self.label_gold = label_gold
        self.deps = deps
        self.label_predicted = label_predicted
        self.tokens = []
        self.feature_generator = None

    def add_token(self, token):
        """

        :param token:
        """
        self.tokens.append(token)

    def set_feature_generator(self, feature_generator):
        """

        :param feature_generator:
        """
        self.feature_generator = feature_generator
        return self

    def get_features(self):
        """

        :return:
        :raise: Exception
        """
        if self.feature_generator is None:
            raise Exception('Feature generator should be set before')
        return self.feature_generator.generate_features()

    def get_text(self):
        return self.text

    def get_deps(self):
        return self.deps

    def get_tokens(self):
        return self.tokens

    def get_length(self):
        return len(self.tokens)

    def get_label_gold(self):
        return self.label_gold

    def set_label_predicted(self, label):
        if self.label_predicted is None:
            self.label_predicted = label
        else:
            raise Exception('Predicted label already set')

    def get_label_predicted(self):
        return self.label_predicted

    def is_labeled_correctly(self):
        return self.get_label_gold() == self.get_label_predicted()

    def generate_deps(self):
        path_to_jar = '/Users/bobrusha/Downloads/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar'
        path_to_models_jar = '/Users/bobrusha/Downloads/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0-models.jar'
        dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
        parse = dependency_parser.raw_parse(self.get_text())
        dep = parse.next()
        # dependencies in instance e.g. [((u'recieved', u'VBD'), u'nsubj', (u'Hailey', u'NNP')),...]
        self.deps = list(dep.triples())
