#!/usr/bin/env python
from Corpus import Corpus
from CorpusReaderAlm import CorpusReaderAlm
from CorpusReaderAman import CorpusReaderAman
from CorpusReaderTweets import CorpusReaderTweets
from CorpusReaderSemeval import CorpusReaderSemeval
from FeatureGeneratorOCC import FeatureGeneratorOCC
from FeatureGeneratorDD import FeatureGeneratorDD
from FeatureGeneratorSRL import FeatureGeneratorSRL
import sys
import pandas as pd

if len(sys.argv) < 6:
    print "Error: Please specify corpus path, model name, path to positive lexicon, path to negative lexicon, path to new csv file"
    sys.exit()
# Define reader
corpus_type = sys.argv[1].split("/")[1]
if corpus_type == "alm":
    reader = CorpusReaderAlm()
elif corpus_type == "alm_trial":
    reader = CorpusReaderAlm()
elif corpus_type == "aman" or corpus_type == "aman_small":
    reader = CorpusReaderAman()
elif corpus_type == "tweets":
    reader = CorpusReaderTweets()
elif corpus_type == "semeval":
    reader = CorpusReaderSemeval()
    xml = sys.argv[6]
else:
    raise Exception('Corpus is unknown')

if corpus_type == "semeval":
    instances, labels = reader.get_instances(sys.argv[1], sys.argv[6])
else:
    instances, labels = reader.get_instances(sys.argv[1])
corpus = Corpus(instances, labels)

pos_lexicon = sys.argv[3]
neg_lexicon = sys.argv[4]
feature_csv = sys.argv[5]
# generate feature csv based on model
if sys.argv[2] == "Majority":
    pass

elif sys.argv[2] in ["OCC", "OCCRB"]:
    for inst in corpus.get_instances():
        inst.generate_deps()
    new_feature_csv = FeatureGeneratorOCC(corpus.get_instances()).generate_feature_csv(feature_csv, pos_lexicon, neg_lexicon)

elif sys.argv[2] == "OCCSRL":
    new_feature_csv = FeatureGeneratorSRL(corpus.get_instances()).generate_feature_csv(feature_csv, pos_lexicon,
                                                                                       neg_lexicon)
elif sys.argv[2] == "DD":
    new_feature_csv, postag_set = FeatureGeneratorDD(corpus.get_instances()).generate_feature_csv(feature_csv, pos_lexicon,neg_lexicon)

elif sys.argv[2] == "COMB":
    occ = FeatureGeneratorOCC(corpus.get_instances()).generate_feature_csv(feature_csv, pos_lexicon, neg_lexicon)
    occ_read = pd.read_csv(occ)
    dd, postag_set = FeatureGeneratorDD(corpus.get_instances()).generate_feature_csv(feature_csv, pos_lexicon,neg_lexicon)
    dd_read = pd.read_csv(dd)
    dd_read.merge(occ_read, how='outer').to_csv(feature_csv, index=False)

else:
    raise Exception('Model is unknown')
