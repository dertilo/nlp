import sys
sys.path.append('.')
from collections import Counter
from pprint import pprint
from typing import Dict

from sklearn.model_selection import ShuffleSplit
import numpy as np

from model_evaluation.classification_metrics import calc_classification_metrics
from model_evaluation.crossvalidation import calc_mean_std_scores
from text_classification.classifiers.common import GenericClassifier
from text_classification.classifiers.embedding_bag_pytorch import EmbeddingBagClassifier
from text_classification.classifiers.tfidf_elasticnet_pytorch import TfIdfElasticNetPytorchClf
from text_classification.classifiers.tfidf_sgd_sklearn import raw_bow, TfIdfSGDSklearnClf
from text_classification.data_readers import get_20newsgroups_data


def benchmark(build_pipeline_fun,parameters:Dict,data):

    def score_fun(train_data, test_data):
        pipeline:GenericClassifier = build_pipeline_fun(**parameters)
        pipeline.fit(train_data)

        proba,y_train = pipeline.predict_proba_encode_targets(train_data)
        pred = np.array(proba == np.expand_dims(np.max(proba, axis=1), 1),dtype='int64')
        target_names = pipeline.target_binarizer.classes_.tolist()
        train_scores = calc_classification_metrics(proba, pred, y_train, target_names=target_names)

        proba,y_test = pipeline.predict_proba_encode_targets(test_data)
        pred = np.array(proba == np.expand_dims(np.max(proba, axis=1), 1),dtype='int64')
        test_scores = calc_classification_metrics(proba, pred, y_test, target_names=target_names)

        return {
            'train-PR-AUC-macro': train_scores['PR-AUC-macro'],
            'test-PR-AUC-macro': test_scores['PR-AUC-macro'],
        }


    splitter = ShuffleSplit(n_splits=1, test_size=0.3, random_state=111)
    splits = [(train, test) for train, test in
              splitter.split(X=range(len(data)))]

    m_scores_std_scores = calc_mean_std_scores(data, score_fun, splits)
    print(build_pipeline_fun.__name__,parameters)
    pprint(m_scores_std_scores['m_scores'])

if __name__ == '__main__':
    from pathlib import Path
    home = str(Path.home())
    data = [{'text':text,'labels':[label]} for text,label in get_20newsgroups_data('train')]
    label_counter = Counter([d['labels'][0] for d in data])
    pprint(label_counter)

    benchmark(TfIdfSGDSklearnClf, parameters={'text_to_bow_fun':raw_bow, 'alpha':0.00002}, data=data)
    benchmark(TfIdfElasticNetPytorchClf,parameters={'text_to_bow_fun':raw_bow,'alpha':0.00002},data=data)
    # benchmark(EmbeddingBagClassifier,parameters={'embedding_dim':9,'alpha':0.0},data=data)
