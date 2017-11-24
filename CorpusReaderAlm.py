# -*- coding: utf-8 -*-
import os
from itertools import izip
from CorpusReader import CorpusReader
from Instance import Instance
from Token import Token



class CorpusReaderAlm(CorpusReader):
    """Use sentences with high annotation agreement, i.e. sentences with four identical emotion labels.
    Five emotions (happy,  fearful,  sad,  surprised and angry-disgusted) from the Ekman’s list of basic emotions were
    used for sentences annotations. Because of data sparsity and related semantics between anger and
    disgust, these two emotions were merged together.
    The Affective Label Codes are: 2=Angry-Disgusted, 3=Fearful, 4=Happy, 6=Sad, 7=Surprised"""
    def get_instances(self, folder):

        instances = []
        labels = set()
        for author in os.listdir(folder):
            path = folder + "/" + author + "/agree-sent/"
            path_pos = folder + "/" + author + "/pos/"
            if os.path.exists(path) and os.path.exists(path_pos):
                for af in os.listdir(path):
                    current = os.path.join(path, af)
                    current_pos = os.path.join(path_pos, af.split('.')[0]+'.sent.okpuncs.props.pos')
                    if os.path.isfile(current) and os.path.isfile(current_pos):
                        agree_data = open(current, "rb")
                        pos_data = open(current_pos, "rb").readlines()
                        for x in agree_data:
                            x = x.strip()
                            id = int(x.split("@")[0])
                            y = pos_data[id].strip()
                            label = int(x.split("@")[1])
                            text = x.split("@")[2]
                            inst = Instance(text, label)
                            for tagtoken in y.split("):("):
                                tag = tagtoken.split(" ")[0].lstrip("(")
                                token = tagtoken.split(" ")[1]
                                token = Token(token, tag)
                                inst.add_token(token)
                            instances.append(inst)
                            labels.add(label)

        return instances, labels
