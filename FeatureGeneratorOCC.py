from __future__ import unicode_literals
import csv
import re
from nltk.stem import WordNetLemmatizer
from FeatureGenerator import FeatureGenerator
from Token import Token
from collections import Counter, defaultdict
import en_core_web_sm


class FeatureGeneratorOCC(FeatureGenerator):
    def __init__(self, instances):
        self.instances = instances


    def generate_feature_csv(self, feature_csv, pos_lexicon, neg_lexicon):
        """
        Generates a csv file with features extracted from instances according to OCC model

        :param feature_csv:
        :param pos_lexicon:
        :param neg_lexicon:
        :return:
        """
        nlp = en_core_web_sm.load()
        first_person_pronouns = set(['i', 'me', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'])
        past_tags = ["VBD", "VBN"]
        future_modals = ["will", "shall", "'ll"]

        with open(feature_csv, 'wb') as f:
            wr = csv.writer(f)
            id = 1
            # feature file header
            wr.writerow(
                ["ID", "text", "direction", "tense", "overall_polarity", "event_polarity", "action_polarity", "label"])

            for inst in self.instances:
                direction = 'other'
                tense = []
                events = {}
                event_started = False
                actions = {}
                action_started = False

                for triple in inst.get_deps():
                    if triple[1] == 'nsubj':
                        # get direction variable
                        pronoun_lower = triple[2][0].lower()
                        if pronoun_lower in first_person_pronouns:
                            direction = 'self'

                        # get action_polarity variable
                        if not action_started:
                            # check next line!!!!!!!!!!!!!!!
                            if triple[0][1].startswith("V") or triple[0][1] == "MD":
                                # get action verb
                                verb = triple[0]
                                actions[verb] = []
                                action_started = True
                                action_last_word = verb
                    if action_started:
                        if len(actions[verb]) == 0:
                            actions[verb] = [verb]
                            heads = [verb]
                        else:
                            if triple[0] in heads:
                                action_last_word = triple[2]
                                actions[verb].append(action_last_word)
                                heads.append(action_last_word)

                    # get tense variable - only for events, others get 'na'
                    if triple[1] == 'dobj':
                        tag = triple[0][1]
                        verb_token = triple[0][0]
                        if tag in past_tags:
                            tense.append("past")
                        if verb_token in future_modals:
                            tense.append("future")
                        else:
                            tense.append("present")

                        if not event_started:
                            # get event
                            obj = triple[2]
                            events[obj] = []
                            event_started = True
                            event_last_word = obj

                    if event_started:
                        if len(events[obj]) == 0:
                            events[obj] = [obj]
                            heads = [obj]
                        else:
                            if triple[0] in heads:
                                event_last_word = triple[2]
                                events[obj].append(event_last_word)
                                heads.append(event_last_word)
                            else:
                                event_started = False

                # print "DIRECTION VAR: ",  direction

                # print "TENSE VAR: ", tense
                if len(tense) == 0:
                    tense = "na"
                else:
                    tense = Counter(tense).most_common(1)[0][0]

                overall_tokens_list = [token for token in inst.get_tokens()]  # tokens as objects
                overall_polarity = self.get_lexicon_features(overall_tokens_list, pos_lexicon, neg_lexicon)
                # print "OVERALL POLARITY VAR: " + overall_polarity
                # print "ACTION VAR: ", actions
                # print "EVENT VAR: ", events

                if len(actions) != 0:
                    actions_polarity = []
                    for action in actions:
                        action_tokens_list = [Token(token, tag) for token, tag in actions[action]]
                        action_polarity = self.get_lexicon_features(action_tokens_list, pos_lexicon, neg_lexicon)
                        actions_polarity.append(action_polarity)
                    final_action_polarity = Counter(actions_polarity).most_common(1)[0][0]
                else:
                    final_action_polarity = "na"
                    # print inst.get_deps()
                    # print "Actions: ", actions
                    # print "Events: ", events
                if len(events) != 0:
                    events_polarity = []
                    for event in events:
                        event_tokens_list = [Token(token, tag) for token, tag in events[event]]
                        event_polarity = self.get_lexicon_features(event_tokens_list, pos_lexicon, neg_lexicon)
                        events_polarity.append(event_polarity)
                    final_event_polarity = Counter(events_polarity).most_common(1)[0][0]
                else:
                    final_event_polarity = "na"

                # wr.writerow(
                #     [id, inst.get_text(), direction, tense, overall_polarity, final_event_polarity,
                #      final_action_polarity, inst.get_label_gold()])
                wr.writerow(
                    [unicode(id).encode("utf-8"),
                     unicode(inst.get_text()).encode("utf-8"),
                     unicode(direction).encode("utf-8"),
                     unicode(tense).encode("utf-8"),
                     unicode(overall_polarity).encode("utf-8"),
                     unicode(final_event_polarity).encode("utf-8"),
                     unicode(final_action_polarity).encode("utf-8"),
                     unicode(inst.get_label_gold()).encode("utf-8")])
                id += 1
        return feature_csv

