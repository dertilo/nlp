import numpy as np
from sklearn.linear_model import SGDClassifier

from text_classification.classifiers.common import GenericClassifier
from text_classification.classifiers.tfidf_dataprocessor import TfIdfTextClfDataProcessor


class TfIdfSGDSklearnClf(GenericClassifier):
    def __init__(self, process_data_to_bows_fun, alpha = 0.00001,
                 get_targets_fun=None
                 ) -> None:
        super().__init__()
        self.dataprocessor = TfIdfTextClfDataProcessor(process_data_to_bows_fun,get_targets_fun=get_targets_fun)
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
