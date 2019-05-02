from collections import Counter
from pprint import pprint
from typing import Dict
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, Binarizer, MultiLabelBinarizer
import re

from text_classification.classifiers.common import GenericClassifier


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

def windowed_bow(data):
    raw_bows = [text_to_bow(d['utterance'])+['SPEAKER__'+d['speaker']] for d in data]

    def window_prefixed_bows(idx,before=-8,after=2):
        return [str(i)+'__'+tok
                for i in range(before,after)
                if idx+i<len(data) and idx+i>=0
                if data[idx+i]['debatefile']==data[idx]['debatefile']
                for tok in raw_bows[idx+i]]

    prefixed_bows = [window_prefixed_bows(idx) for idx in range(len(data))]
    return prefixed_bows

def raw_bow(texts):
    raw_bows = [text_to_bow(text) for text in texts]
    return raw_bows

class TfIdfSGDSklearnClf(GenericClassifier):
    def __init__(self, text_to_bow_fun, alpha = 0.00001) -> None:
        super().__init__()
        self.text_to_bow_fun = text_to_bow_fun
        vectorizer = TfidfVectorizer(sublinear_tf=True,
                                     preprocessor=identity_dummy_method,
                                     tokenizer=identity_dummy_method,
                                     ngram_range=(1, 1),
                                     max_df=0.75, min_df=2,
                                     max_features=30000,
                                     stop_words=None  # 'english'
                                     )
        clf = SGDClassifier(alpha=alpha, loss='log', penalty="elasticnet", l1_ratio=0.2, tol=1e-3)
        self.pipeline = Pipeline([('tfidf', vectorizer), ('clf', clf)])

    def fit(self,X,y=None):
        self.target_binarizer = MultiLabelBinarizer()
        self.target_binarizer.fit([d['labels'] for d in X])
        y_train_bin = self.target_binarizer.transform([d['labels'] for d in X])
        self.pipeline.fit(self.text_to_bow_fun([d['text'] for d in X]), np.argmax(y_train_bin, axis=1))
        assert len(self.target_binarizer.classes_) > 1
        return self

    def predict_proba(self,X):
        return self.pipeline.predict_proba(self.text_to_bow_fun([d['text'] for d in X]))

    def predict_proba_encode_targets(self, data):
        probas = self.predict_proba(data)
        targets = self.target_binarizer.transform([d['labels'] for d in data]).astype('int64')
        return probas, targets
