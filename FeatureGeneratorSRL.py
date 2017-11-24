from __future__ import unicode_literals
import csv
import re
from nltk.stem import WordNetLemmatizer
from FeatureGenerator import FeatureGenerator
from Token import Token
from collections import Counter, defaultdict
import en_core_web_sm
from practnlptools.tools import Annotator
from nltk import word_tokenize


class FeatureGeneratorSRL(FeatureGenerator):
    def __init__(self, instances):
        self.instances = instances
        self.annotator = Annotator() # SRL tool

    def generate_feature_csv(self, feature_csv, pos_lexicon, neg_lexicon):
        """
        Generates a csv file with SRL features extracted from instances

        :param feature_csv:
        :param pos_lexicon:
        :param neg_lexicon:
        :return:
        """
        first_person_pronouns = set(['i', 'me', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'])
        past_tags = ["VBD", "VBN"]
        future_modals = ["will", "shall", "'ll"]

        with open(feature_csv, 'wb') as f:
            srlabels = set()
            wr = csv.writer(f)
            id = 1
            # feature file header
            wr.writerow(
                ["ID", "text", "direction", "tense", "overall_polarity", "event_polarity", "action_polarity",
                 "label"])

            for inst in self.instances:
                direction = 'other' # default
                tense = []
                events_polarity = []
                actions_polarity = []

                inst_analysed = self.annotator.getAnnotations(inst.get_text().replace("\"", "'").encode('utf-8'))
                srl_dict_list = inst_analysed['srl']
                pos_tuple_list = inst_analysed['pos']
                if len(srl_dict_list) > 0:
                    for srl_dict in srl_dict_list:
                        A0 = srl_dict.get('A0')
                        if A0 is not None:
                            A0_lower_tokenized = [token.lower() for token in word_tokenize(A0)]
                            if len(set(A0_lower_tokenized).intersection(first_person_pronouns)) > 0:
                                direction = 'self'

                        action_tokens_list = []
                        verb = srl_dict.get('V').split(" ")[0] # in case of such verbs as 'hanging out' - take only 1st word
                        tuples = [item for item in pos_tuple_list if item[0] == verb]
                        if len(tuples) > 0:
                            verb_tag = tuples[0][1] # find tag for the verb
                        action_tokens_list.append(Token(verb, verb_tag))
                        if verb_tag in past_tags:
                            tense.append("past")
                        elif 'AM-MOD' in srl_dict and srl_dict['AM-MOD'] in future_modals:
                            tense.append("future")
                        else:
                            tense.append("present")

                        event = srl_dict.get("A1")
                        if event is not None:
                            event_tokens_list = []
                            event_lower_tokenized = [token.lower() for token in word_tokenize(event)]
                            for token in event_lower_tokenized:
                                tuples = [item for item in pos_tuple_list if item[0].lower() == token]
                                if len(tuples) > 0:
                                    tag = tuples[0][1]
                                    event_tokens_list.append(Token(token, tag))
                            event_polarity = self.get_lexicon_features(event_tokens_list, pos_lexicon, neg_lexicon)
                            events_polarity.append(event_polarity)

                        # can be several modifiers within one srl_dict
                        actions = [v for k, v in srl_dict.items() if k.startswith("AM-")]
                        if len(actions) > 0:
                            for action in actions:
                                action_lower_tokenized = [token.lower() for token in word_tokenize(action)]
                                for token in action_lower_tokenized:
                                    tuples = [item for item in pos_tuple_list if item[0].lower() == token]
                                    if len(tuples) > 0:
                                        tag = tuples[0][1]
                                    action_tokens_list.append(Token(token, tag))
                            action_polarity = self.get_lexicon_features(action_tokens_list, pos_lexicon, neg_lexicon)
                            actions_polarity.append(action_polarity)

                if len(events_polarity) > 0:
                    final_event_polarity = Counter(events_polarity).most_common(1)[0][0]
                else:
                    final_event_polarity = "na"

                if len(actions_polarity) > 0:
                    final_action_polarity = Counter(actions_polarity).most_common(1)[0][0]
                else:
                    final_action_polarity = "na"

                if len(tense) > 0:
                    final_tense = Counter(tense).most_common(1)[0][0]
                else:
                    final_tense = "na"

                overall_tokens_list = [token for token in inst.get_tokens()]  # tokens as objects
                overall_polarity = self.get_lexicon_features(overall_tokens_list, pos_lexicon, neg_lexicon)

                # wr.writerow(
                #     [id, inst.get_text(), direction, tense, overall_polarity, final_event_polarity,
                #      final_action_polarity, inst.get_label_gold()])
                wr.writerow(
                    [unicode(id).encode("utf-8"),
                     unicode(inst.get_text()).encode("utf-8"),
                     unicode(direction).encode("utf-8"),
                     unicode(final_tense).encode("utf-8"),
                     unicode(overall_polarity).encode("utf-8"),
                     unicode(final_event_polarity).encode("utf-8"),
                     unicode(final_action_polarity).encode("utf-8"),
                     unicode(inst.get_label_gold()).encode("utf-8")])
                id += 1

        return feature_csv

