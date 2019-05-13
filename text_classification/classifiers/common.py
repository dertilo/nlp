import abc
from abc import abstractmethod


class GenericClassifier(object):
    target_binarizer = None

    @abc.abstractmethod
    def fit(self, data):
        raise NotImplementedError

    @abc.abstractmethod
    def predict_proba(self, data):
        raise NotImplementedError

    @abc.abstractmethod
    def predict_proba_encode_targets(self, data):
        raise NotImplementedError


class DataProcessorInterface(object):
    target_binarizer = None

    @abstractmethod
    def fit(self, data):
        raise NotImplementedError

    @abstractmethod
    def transform(self,data):
        raise NotImplementedError

    @abstractmethod
    def build_get_batch_fun(self,raw_data,batch_size):
        raise NotImplementedError

    @property
    def num_classes(self):
        return self.target_binarizer.classes_.shape[0]