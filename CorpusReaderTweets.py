# -*- coding: utf-8 -*-
from nltk import pos_tag, word_tokenize
from CorpusReader import CorpusReader
from Instance import Instance
from Token import Token
from nltk.tag.perceptron import PerceptronTagger
import re
import io

class CorpusReaderTweets(CorpusReader):
    """e.g. cleaning twitter data may involve :
    1. Extracting tweet text, hashtags, tweet dates and usernames from tweet objects
    2. Cleaning the tweet text by removing usernames, urls and hashtags(replace #hashtag by hashtag) and removing stop words


    replaced letters/punctuations that are re- peated more than twice with the same two letters/punctuations (e.g., cooool → cool, !!!!! → !!);
    normalized some frequently used informal expressions (e.g., ll → will, dnt → do not);
    stripped hash symbols (#tomorrow → tomorrow).
    """
    def get_instances(self, folder):
        instances = []
        labels = set()
        tagger = PerceptronTagger() # load nltk perceptron just once to speed up tagging
        with io.open(folder, encoding="utf-8") as f:
        # with open(folder) as f:
            for line in f:
                # line = unicode(line).encode("utf-8")
                line_split = line.rstrip().split("\t")
                if len(line_split) != 3:
                    continue
                id, text, label = line_split
                id = id.rstrip(":")
                text = re.sub('[#]', '', text.rstrip())
                label = re.sub('[^a-z]', '', label)
                inst = Instance(text, label)
                inst_tokenized = word_tokenize(text)
                inst_tagged = tagger.tag(inst_tokenized)
                for tokentag in inst_tagged:
                    token = Token(tokentag[0], tokentag[1])
                    inst.add_token(token)
                instances.append(inst)
                labels.add(label)
        return instances, labels
