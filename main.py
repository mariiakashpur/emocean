#!/usr/bin/env python

from ModelDD import ModelDD
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import scipy as sp
import sys

# Define feature file and train/val/test sets
data_file = sys.argv[1]
df = pd.read_csv(data_file)
df = df.drop(['ID'], axis=1)
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])
# print df.head(10)
# print encoder.classes_

# Get BOW using CountVectorizer
def get_bow(X_train, X_test):
    # TfidfVectorizer=CountVectorizer+TfidfTransformer; stop_words="english" as another optional argument
    tfidf_vect = TfidfVectorizer(binary=True, ngram_range=(1, 2))
    X_train_sp = sp.sparse.hstack((tfidf_vect.fit_transform(X_train['text']), X_train.drop(['text'], axis=1).values), format='csr')
    X_test_sp = sp.sparse.hstack((tfidf_vect.transform(X_test['text']), X_test.drop(['text'], axis=1).values), format='csr')
    return X_train_sp, X_test_sp


X = df.drop(['ID', 'label'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train_sp, X_test_sp = get_bow(X_train, X_test)
model = ModelDD(X_train_sp, y_train, X_test_sp, y_test)



print "CROSS-VALIDATION RESULTS\n"
model.predict_cv()
model.evaluate_cv()


print "TEST SET RESULTS\n"
model.predict_test()
model.evaluate_test()
