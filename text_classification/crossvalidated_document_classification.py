from collections import Counter
from pprint import pprint

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelEncoder, Binarizer, MultiLabelBinarizer

from model_evaluation.classification_metrics import calc_classification_metrics
from model_evaluation.crossvalidation import calc_mean_std_scores
from text_classification.data_readers import get_20newsgroups_data, get_GermEval2017_TaskB_data
import re


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
    # data_test = get_20newsgroups_data('test')

    # def convert_GermEval2017_data(d):
    #     return d['text'],d['sentiment']
    #
    # data_train = [convert_GermEval2017_data(d) for d in get_GermEval2017_TaskB_data('../data/train_v1.4.tsv')]
    # # data_test = [convert_GermEval2017_data(d) for d in get_GermEval2017_TaskB_data('../data/test_TIMESTAMP1.tsv')]


    pprint(Counter([label for _, label in data_train]))

    label_encoder = MultiLabelBinarizer()
    label_encoder.fit([[label] for _, label in data_train])

    def score_fun(train_data,test_data):
        vectorizer = TfidfVectorizer(sublinear_tf=True,
                                     preprocessor=identity_dummy_method,
                                     tokenizer=identity_dummy_method,
                                     ngram_range=(1, 1),
                                     max_df=0.75, min_df=2,
                                     max_features=30000,
                                     stop_words=None  # 'english'
                                     )

        X_train = vectorizer.fit_transform([text_to_bow(text) for text, _ in train_data])
        X_test = vectorizer.transform([text_to_bow(text) for text, _ in test_data])

        clf = SGDClassifier(alpha=.00001, loss='log', penalty="elasticnet", l1_ratio=0.2,tol=1e-3)
        y_train = label_encoder.transform([[label] for _, label in train_data])
        y_test = label_encoder.transform([[label] for _, label in test_data])
        clf.fit(X_train, np.argmax(y_train,axis=1))

        proba = clf.predict_proba(X_train)
        pred = np.array(proba == np.expand_dims(np.max(proba, axis=1), 1),dtype='int64')
        target_names = label_encoder.classes_.tolist()
        train_scores = calc_classification_metrics(proba, pred, y_train, target_names=target_names)

        proba = clf.predict_proba(X_test)
        pred = np.array(proba == np.expand_dims(np.max(proba, axis=1), 1),dtype='int64')

        test_scores = calc_classification_metrics(proba, pred, y_test, target_names=target_names)

        return {
            'train':train_scores,
            'test':test_scores,
        }

    splitter = ShuffleSplit(n_splits=5, test_size=0.2, random_state=111)
    splits = [(train,test) for train,test in splitter.split(X=range(len(data_train)))]

    m_scores_std_scores = calc_mean_std_scores(data_train, score_fun, splits)
    pprint(m_scores_std_scores['m_scores']['test'])

