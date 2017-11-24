from Classifier import Classifier
from collections import Counter
from sklearn.preprocessing import LabelEncoder

class ClassifierRB(Classifier):
    def __init__(self, label_encoder):
        super(ClassifierRB, self).__init__()
        self.label_encoder = label_encoder

    def predict(self, feature_rows):
        predicted_labels = []
        csv_most_frequent = Counter(feature_rows['label'].tolist()).most_common(1)[0][0]
        # csv_most_frequent = "joy" # alm corpus
        for index, row in feature_rows.iterrows():
            ID = row['ID']
            text = row['text']
            direction = row['direction']
            tense = row['tense']
            overall_polarity = row['overall_polarity']
            event_polarity = row['event_polarity']
            action_polarity = row['action_polarity']
            predicted_label_action = None
            predicted_label_event = None

            label_mapping = {"fear":"fear", "fear-confirmed":"fear","joy":"joy", "happy-for":"joy", "satisfaction":"joy",
                             "admiration":"joy", "pride":"joy","anger":"anger", "reproach":"anger", "distress":"sadness",
                             "sorry-for":"sadness", "disappointment":"sadness", "shame":"sadness", "resentment":"disgust",
                             "hope":"joy", "relief":"joy", "gloating":"joy"}
            # event exists
            if event_polarity in ["positive", "negative"]:
                if overall_polarity in ["positive", "negative"]:
                    if tense != "na":
                        if direction == "self":
                            if tense == "future" and overall_polarity == "positive" and event_polarity == "positive":
                                predicted_label_event = label_mapping["hope"]
                            elif tense == "future" and overall_polarity == "negative" and event_polarity == "negative":
                                predicted_label_event = label_mapping["fear"]
                            elif tense == "present" and overall_polarity == "positive" and event_polarity == "positive":
                                predicted_label_event = label_mapping["joy"]
                            elif tense == "present" and overall_polarity == "negative" and event_polarity == "negative":
                                predicted_label_event = label_mapping["distress"]
                            elif tense == "past" and overall_polarity == "positive" and event_polarity == "positive":
                                predicted_label_event = label_mapping["satisfaction"]
                            elif tense == "past" and overall_polarity == "negative" and event_polarity == "negative":
                                predicted_label_event = label_mapping["fear-confirmed"]
                            elif tense == "past" and overall_polarity == "positive" and event_polarity == "negative":
                                predicted_label_event = label_mapping["relief"]
                            elif tense == "past" and overall_polarity == "negative" and event_polarity == "positive":
                                predicted_label_event = label_mapping["disappointment"]
                        else:
                            if overall_polarity == "positive" and event_polarity == "positive":
                                predicted_label_event = label_mapping["happy-for"]
                            elif overall_polarity == "negative" and event_polarity == "positive":
                                predicted_label_event = label_mapping["resentment"]
                            elif overall_polarity == "positive" and event_polarity == "negative":
                                predicted_label_event = label_mapping["gloating"]
                            elif overall_polarity == "negative" and event_polarity == "negative":
                                predicted_label_event = label_mapping["sorry-for"]
            # action exists
            if action_polarity in ["positive", "negative"]:
                if direction == "self":
                    if action_polarity == "positive":
                        predicted_label_action = label_mapping["pride"]
                    else:
                        predicted_label_action = label_mapping["shame"]
                else:
                    if action_polarity == "positive":
                        predicted_label_action = label_mapping["admiration"]
                    else:
                        predicted_label_action = label_mapping["reproach"]

            if predicted_label_action is not None and predicted_label_event is not None:
                predicted_label = predicted_label_action
                print "both not none: " + text + "overall pol: " + overall_polarity + " event pol: " + event_polarity + " act pol: " + action_polarity +  " dir: " + direction + " tense: " + tense
            else:
                if predicted_label_action:
                    predicted_label = predicted_label_action
                    print "only action: " + text + "overall pol: " + overall_polarity +  "event pol: " + event_polarity + " act pol: " + action_polarity + " dir: " + direction + " tense: " + tense
                elif predicted_label_event:
                    predicted_label = predicted_label_event
                    print "only event: " + text + "overall pol: " + overall_polarity + " event pol: " + event_polarity + " act pol: " + action_polarity + " dir: " + direction + " tense: " + tense
                else:
                    predicted_label = csv_most_frequent
                    print "no action and event: " + text + "overall pol: " + overall_polarity
            predicted_labels.append(predicted_label)

        # workaround for alm corpus & label encoder
        # # @todo fix alm corpus mappings - since I use encoder, it encodes digital lables in alm into other digits [0 1 2 3 4] - [2 3 4 6 7]
        # mapping_alm_encoder1 = {4: "joy"}
        # predicted_labels = [mapping_alm_encoder1.get(item, item) for item in predicted_labels]
        # mapping_alm_encoder2 = {"fear":1, "joy":2, "anger":0, "sadness":3, "disgust":0}
        # predicted_labels_mapped = [mapping_alm_encoder2.get(item, item) for item in predicted_labels]
        # return predicted_labels_mapped

        return self.label_encoder.transform(predicted_labels)




