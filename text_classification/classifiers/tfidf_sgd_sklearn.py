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

from text_classification.classifiers.common import GenericClassifier
from text_classification.classifiers.tfidf_dataprocessor import TfIdfTextClfDataProcessor


class TfIdfSGDSklearnClf(GenericClassifier):
    def __init__(self, text_to_bow_fun, alpha = 0.00001) -> None:
        super().__init__()
        self.text_to_bow_fun = text_to_bow_fun
        self.dataprocessor = TfIdfTextClfDataProcessor(text_to_bow_fun)
        self.clf = SGDClassifier(alpha=alpha, loss='log', penalty="elasticnet", l1_ratio=0.2, tol=1e-3)

    def fit(self,X,y=None):
        self.dataprocessor.fit(X)
        inputs, targets = self.dataprocessor.process_inputs_and_targets(X)
        self.clf.fit(inputs, np.argmax(targets, axis=1))
        assert len(self.dataprocessor.target_binarizer.classes_) > 1
        return self

    def predict_proba(self,X):
        inputs = self.dataprocessor.process_inputs(X)
        return self.clf.predict_proba(inputs)

    def predict_proba_encode_targets(self, data):
        inputs, targets = self.dataprocessor.process_inputs_and_targets(data)
        probas = self.clf.predict_proba(inputs)
        return probas, targets.astype('int64')

    @property
    def target_binarizer(self):
        return self.dataprocessor.target_binarizer
