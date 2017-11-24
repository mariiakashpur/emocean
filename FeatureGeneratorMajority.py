from __future__ import division
from FeatureGenerator import FeatureGenerator
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import csv
import scipy as sp
import pandas as pd
import numpy as np
from collections import Counter
from Corpus import Corpus

class FeatureGeneratorDD(FeatureGenerator):
    # def __init__(self, instance):
    #     self.instance = instance

    def __init__(self, instances):
        """

        :param instances: it`s possible to have test|validation|train instances here
        """
        self.instances = instances

    def generate_features(self):
        vectorizer = CountVectorizer(min_df=1)
        print self.instances
        X = vectorizer.fit_transform(self.instances)
        return X, vectorizer

    def generate_feature_csv(self, feature_csv, pos_lexicon, neg_lexicon, postag_instances=None):
        if postag_instances:
            corpus_postag_set = Corpus.get_postag_set(postag_instances) # return all tags in corpus in a list
        else:
            corpus_postag_set = Corpus.get_postag_set(self.instances) # return all tags in corpus in a list

        # ID, text, pos_feature, neg_feature, percentages for all corpus tags, label
        with open(feature_csv, 'wb') as f:
            # wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr = csv.writer(f)
            id = 1
            wr.writerow(["ID", "text", "pos", "neg"]+corpus_postag_set+["label"])
            for inst in self.instances:
                inst_postags = [token.get_tag() for token in inst.get_tokens()]
                inst_postag_counter = Counter(inst_postags)
                postag_percent = []
                for tag in corpus_postag_set:
                    if tag in inst_postag_counter:
                        # percentage of words belonging to each POS in instance
                        postag_percent.append(inst_postag_counter[tag]/inst.get_length())
                    else:
                        postag_percent.append(0)
                pos_neg_list = self.get_lexicon_features(inst.get_text(), pos_lexicon, neg_lexicon)
                wr.writerow([id, inst.get_text(), pos_neg_list[0], pos_neg_list[1]]+postag_percent+[inst.get_label_gold()])
                id += 1
        return feature_csv, corpus_postag_set


    def generate_combined_features(self, feature_csv):
        feature_rows = pd.read_csv(feature_csv)
        # Create vectorizer for function to use
        vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
        y = feature_rows["label"].values.astype(np.float32)

        X = sp.sparse.hstack(
            (vectorizer.fit_transform(feature_rows.text), feature_rows[['pos', 'neg']+Corpus.get_postag_set(self.instances)].values),
            format='csr'
        )
        return X, y, vectorizer

    def generate_combined_features_test(self, feature_csv, vectorizer, postag_set):
        """ Pass vectorizer used for training as argument"""
        feature_rows = pd.read_csv(feature_csv)
        y = feature_rows["label"].values.astype(np.float32)
        X = sp.sparse.hstack((vectorizer.transform(feature_rows.text), feature_rows[['pos', 'neg']+[tag for tag in postag_set]].values),
                             format='csr')
        return X, y



