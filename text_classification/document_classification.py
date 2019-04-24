from collections import Counter
from pprint import pprint
from time import time

import numpy as np
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from text_classification.data_readers import get_20newsgroups_data, get_GermEval2017_TaskB_data
import re


def encode_targets(data_train,data_test):
    train_labels = [label for _, label in data_train]
    test_labels = [label for _, label in data_test]
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)
    y_train = label_encoder.transform(train_labels)
    y_test = label_encoder.transform(test_labels)
    return label_encoder,y_train,y_test

def benchmark(clf):
    print('_' * 80)
    # print(clf)
    clf_descr = str(clf).split('(')[0]
    print(clf_descr)
    t0 = time()

    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred_train = clf.predict(X_train)
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_train, pred_train)
    print("train-f1-micro:   %0.3f" % score)
    score = metrics.accuracy_score(y_test, pred)
    print("test-f1-micro:   %0.3f" % score)

    return {'clf-name':clf_descr,
            'accuracy':np.round(score,2),
            'train-time':np.round(train_time,3),
            'test-time':np.round(test_time,3)}


def identity_dummy_method(x):
    '''
    just to fool the scikit-learn vectorizer
    '''
    return x

def get_nrams(string, min_n=3, max_n=5):
    return [string[k:k + ngs] for ngs in range(min_n, max_n + 1) for k in range(len(string) - ngs)]

def regex_tokenizer(text, pattern=r"(?u)\b\w\w+\b"):# pattern stolen from scikit-learn
    return [m.group() for m in re.finditer(pattern, text)]

def text_to_bow(text):
    return regex_tokenizer(text)

if __name__ == '__main__':
    data_train = get_20newsgroups_data('train')
    data_test = get_20newsgroups_data('test')

    # def convert_GermEval2017_data(d):
    #     return d['text'],d['sentiment']
    #
    # data_train = [convert_GermEval2017_data(d) for d in get_GermEval2017_TaskB_data('../data/train_v1.4.tsv')]
    # data_test = [convert_GermEval2017_data(d) for d in get_GermEval2017_TaskB_data('../data/test_TIMESTAMP1.tsv')]

    vectorizer = TfidfVectorizer(sublinear_tf=True,
                                 preprocessor=identity_dummy_method,
                                 tokenizer=identity_dummy_method,
                                 ngram_range=(1, 1),
                                 max_df=0.75,
                                 min_df=2,
                                 max_features=30000,
                                 stop_words=None#'english'
                                 )
    X_train = vectorizer.fit_transform([text_to_bow(text) for text, _ in data_train])
    print("n_samples: %d, n_features: %d" % X_train.shape)

    X_test = vectorizer.transform([text_to_bow(text) for text, _ in data_test])
    print("n_samples: %d, n_features: %d" % X_test.shape)

    pprint(Counter([label for _, label in data_train]))

    label_encoder, y_train, y_test = encode_targets(data_train,data_test)

    benchmark(SGDClassifier(alpha=.00001, loss='log', penalty="elasticnet", l1_ratio=0.2))

    benchmark(MultinomialNB(alpha=.03))
