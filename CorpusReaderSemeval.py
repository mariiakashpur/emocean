# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
from itertools import izip
from CorpusReader import CorpusReader
from Instance import Instance
from Token import Token
from nltk import pos_tag, word_tokenize
from nltk.tag.perceptron import PerceptronTagger



class CorpusReaderSemeval(CorpusReader):
    """From Semeval data, choose emotion that receives highest score"""

    def get_instances(self, label_file, xml_file):
        instances = []
        labels_final = set()
        tagger = PerceptronTagger()  # load nltk perceptron just once to speed up tagging
        labels_dict = {0: "anger", 1: "disgust", 2: "fear", 3: "joy", 4: "sadness", 5: "surprise"}
        tree = ET.parse(xml_file)
        root = tree.getroot()
        with open(label_file) as f:
            for sent, line in izip(root, f):
                id_xml = sent.attrib.values()[0]
                id_labels = line.rstrip().split()
                id_file = id_labels[0]
                if id_xml == id_file:
                    for i in sent.itertext():
                        text = i
                    labels = id_labels[1:]
                    label = labels.index(str(max([int(label) for label in labels])))
                    inst = Instance(text, labels_dict[label])
                    inst_tokenized = word_tokenize(text)
                    inst_tagged = tagger.tag(inst_tokenized)
                    for tokentag in inst_tagged:
                        token = Token(tokentag[0], tokentag[1])
                        inst.add_token(token)
                    instances.append(inst)
                    labels_final.add(label)
            return instances, labels_final
