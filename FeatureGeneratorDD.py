# -*- coding: utf-8 -*-
from __future__ import division
from FeatureGenerator import FeatureGenerator
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
import csv
import scipy as sp
import pandas as pd
import numpy as np
from collections import Counter
from Corpus import Corpus

class FeatureGeneratorDD(FeatureGenerator):

    def __init__(self, instances):
        """

        :param instances: it`s possible to have test|validation|train instances here
        """
        self.instances = instances

    def get_lexicon_features(self, inst_tokens, pos_lexicon, neg_lexicon):
        lemmatizer = WordNetLemmatizer()
        pos_words = []
        neg_words = []
        with open(pos_lexicon) as pf:
            for line in pf:
                if re.match("[^;]", line) and len(line) > 1:
                    pos_words.append(line.strip())
        with open(neg_lexicon) as nf:
            for line in nf:
                if re.match("[^;]", line) and len(line) > 1:
                    neg_words.append(line.strip())
        pos_neg_list = []
        pos_len = 0
        neg_len = 0
        inst_len = 0
        for token in inst_tokens:
            word = token.get_text()
            tag = token.get_tag()
            wordnet_tag = FeatureGenerator.get_wordnet_pos(tag)
            if wordnet_tag:
                word = lemmatizer.lemmatize(word, wordnet_tag).encode('utf-8')
                # word = str(lemmatizer.lemmatize(word, wordnet_tag))
            else:
                word = lemmatizer.lemmatize(word).encode('utf-8')
                # word = str(lemmatizer.lemmatize(word))
            if word in pos_words:
                pos_len += 1
            elif word in neg_words:
                neg_len += 1
            inst_len += 1
        pos_neg_list.append(pos_len/inst_len)
        pos_neg_list.append(neg_len/inst_len)

        return pos_neg_list


    def generate_feature_csv(self, feature_csv, pos_lexicon, neg_lexicon, postag_instances=None):
        """
         Generates a csv file with features extracted from instances according to data-driven DD model
        :param feature_csv:
        :param pos_lexicon:
        :param neg_lexicon:
        :param postag_instances:
        :return:
        """
        if postag_instances:
            corpus_postag_set = Corpus.get_postag_set(postag_instances) # return all tags in corpus in a list
        else:
            corpus_postag_set = Corpus.get_postag_set(self.instances) # return all tags in corpus in a list

        # feature file header: ID, text, pos_feature, neg_feature, percentages for all corpus tags, label
        with open(feature_csv, 'wb') as f:
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
                # tokens_list = [token.get_text() for token in inst.get_tokens()]
                tokens_list = [token for token in inst.get_tokens()] # tokens as objects
                pos_neg_list = self.get_lexicon_features(tokens_list, pos_lexicon, neg_lexicon)
                # wr.writerow([id, inst.get_text(), pos_neg_list[0], pos_neg_list[1]]+postag_percent+[inst.get_label_gold()])
                wr.writerow(
                    [unicode(id).encode("utf-8"),
                     unicode(inst.get_text()).encode("utf-8"),
                     unicode(pos_neg_list[0]).encode("utf-8"),
                     unicode(pos_neg_list[1]).encode("utf-8")]
                    + postag_percent
                    + [unicode(inst.get_label_gold()).encode("utf-8")])
                id += 1
        return feature_csv, corpus_postag_set


    # seems that it's not used anymore
    def generate_combined_features(self, feature_csv):
        feature_rows = pd.read_csv(feature_csv) # pandas Data Frame object
        # Create vectorizer for function to use
        vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2)) # CountVectorizer constructs BOW model based on word counts
        y = feature_rows["label"].values.astype(np.float32)
        # combine BOW model from Count Vectorizer with self-extracted features
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



