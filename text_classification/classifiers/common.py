import abc


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
