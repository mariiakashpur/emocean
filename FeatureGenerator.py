import abc
import re
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

class FeatureGenerator(object):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        """Convert Penn tagset to Wordnet tagset to use in NLTK lemmatizer as 2nd argument to improve results"""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

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
        pos_neg_dict = defaultdict(int)
        for token in inst_tokens:
            word = token.get_text()
            tag = token.get_tag()
            wordnet_tag = FeatureGenerator.get_wordnet_pos(tag)

            if wordnet_tag:
                # word = str(lemmatizer.lemmatize(word, wordnet_tag))
                word = lemmatizer.lemmatize(word, wordnet_tag).encode('utf-8')
            else:
                # word = str(lemmatizer.lemmatize(word))
                word = lemmatizer.lemmatize(word).encode('utf-8')
            if word in pos_words:
                pos_neg_dict["positive"] += 1
            elif word in neg_words:
                pos_neg_dict["negative"] += 1
        if len(pos_neg_dict) == 0:
            polarity = "neut"
        else:
            polarity = max(pos_neg_dict.iterkeys(), key=(lambda key: pos_neg_dict[key]))
        return polarity

