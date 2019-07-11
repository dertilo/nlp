import json
from collections import Counter
from pprint import pprint
from time import time
from typing import List, Dict, Callable

import hyperopt
import numpy as np
from hyperopt import Trials, STATUS_OK, fmin, tpe, hp
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from model_evaluation.classification_metrics import calc_classification_metrics
from model_evaluation.crossvalidation import calc_mean_std_scores, calc_mean_and_std
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


def initialize_globals(worker_id, dataset_builder):
    #     torch.manual_seed(12345)
    global global_dataset
    global_dataset = dataset_builder(worker_id)


# needs pip install networkx==1.11!! TODO(tilo): why where?
def tune_hyperparams(
        init_fun,
                     search_space:Dict,
                     score_fun,
                     max_evals = 9,
                     trials = Trials(),
                    metric_for_hyperopt='accuracy',
                     n_jobs=5
    ):
    initialize_globals(None,init_fun)
    def objective(params_dict):
        start = time()
        metrics = score_fun(**params_dict)
        score = metrics[metric_for_hyperopt]
        return {'loss': -score,
                'status': STATUS_OK,
                'mean-metrics':metrics,
                'crossval_duration_in_s': time()-start,
                'hyperparams': params_dict
                }

    # start = time()
    _ = fmin(objective, search_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    # duration = time() -start
    min_loss,best_hyperparams = min([(t['loss'],t['hyperparams']) for t in trials.results],key=lambda x:x[0])
    return best_hyperparams,trials

def score_tfidf_classifier(alpha=0.0001,loss='log',penalty='elasticnet',l1_ratio=0.15):
    vectorizer = TfidfVectorizer(sublinear_tf=True,
                                 preprocessor=identity_dummy_method,
                                 tokenizer=identity_dummy_method,
                                 ngram_range=(1, 1),
                                 max_df=0.75,
                                 min_df=2,
                                 max_features=30000,
                                 stop_words=None#'english'
                                 )
    data_train = global_dataset

    vectorized_data = vectorizer.fit_transform([text_to_bow(text) for text, _ in data_train])

    train_labels = [label for _, label in data_train]
    label_encoder = MultiLabelBinarizer()
    targets = label_encoder.fit_transform([[l] for l in train_labels])

    def score_split(train_idx,test_idx):

        X_train = vectorized_data[np.array(train_idx),:]
        y_train = targets[np.array(train_idx),:]

        X_test = vectorized_data[np.array(test_idx),:]
        y_test = targets[np.array(test_idx),:]

        clf = SGDClassifier(alpha=alpha, loss=loss, penalty=penalty, l1_ratio=l1_ratio)
        clf_descr = str(clf).split('(')[0]

        clf.fit(X_train, np.argmax(y_train,axis=1))

        # pred_train = clf.predict(X_train)
        proba = clf.predict_proba(X_test)
        pred = clf.predict(X_test)
        pred_onehot = np.zeros_like(proba,dtype='int64')
        pred_onehot[np.arange(pred_onehot.shape[0]), pred] = 1
        cls_metrics = calc_classification_metrics(proba, pred_onehot, y_test, label_encoder.classes_.tolist())
        return cls_metrics

    splitter = ShuffleSplit(n_splits=2, test_size=0.2, random_state=111)
    splits = [(train,test) for train,test in splitter.split(X=range(vectorized_data.shape[0]))]


    scores = [score_split(train,test) for train,test in splits]
    m_scores, std_scores = calc_mean_and_std(scores)
    return m_scores


if __name__ == '__main__':
    space = {
        'alpha': hp.loguniform('alpha', np.log(10) * -6, np.log(10) * -2),
        'l1_ratio': hp.loguniform('l1_ratio', np.log(10) * -6, 0),
    }

    init_fun = lambda args: get_20newsgroups_data('train')

    # initialize_globals(None,init_fun)
    # print(score_tfidf_classifier())
    best,trials = tune_hyperparams(
        init_fun=init_fun,
        max_evals=2,
        score_fun=score_tfidf_classifier,
        search_space=space,
        trials=Trials()
    )

    for t in trials.results:
        print('loss: %0.2f; f1-micro: %0.2f; params: %s'%(t['loss'],t['mean-metrics']['f1-micro'], json.dumps(t['hyperparams'])))
    print(best)




