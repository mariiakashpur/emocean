# -*- coding: utf-8 -*-
from nltk import pos_tag, word_tokenize
from CorpusReader import CorpusReader
from Instance import Instance
from Token import Token
from nltk.tag.perceptron import PerceptronTagger

class CorpusReaderAman(CorpusReader):
    """Use blogpost sentences for which the annotators agreed on the emotion category."""
    def get_instances(self, folder):
        # happiness/joy???????????????????????????
        labels_dict = {"hp":"joy", "sd":"sadness", "ag":"anger", "dg":"disgust", "sp":"surprise", "fr":"fear"}
        instances = []
        labels = set()
        tagger = PerceptronTagger() # load nltk perceptron just once to speed up tagging
        with open(folder) as f:
            for line in f:
                label, id, text = line.strip().split(" ", 2) # split by first two spaces only
                if label == "ne": # ignore no emotion
                    continue
                inst = Instance(text, labels_dict[label])
                inst_tokenized = word_tokenize(text)
                inst_tagged = tagger.tag(inst_tokenized)
                for tokentag in inst_tagged:
                    token = Token(tokentag[0], tokentag[1])
                    inst.add_token(token)
                instances.append(inst)
                labels.add(label)
        return instances, labels
