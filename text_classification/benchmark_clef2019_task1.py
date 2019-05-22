import sys
sys.path.append('.')
from scipy.sparse import csr_matrix #TODO(tilo): if not imported before torch it throws: ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found

from collections import Counter
from pprint import pprint
from typing import Dict

import numpy
from sklearn.model_selection import GroupShuffleSplit

from getting_data.clef2019 import get_Clef2019_data
from model_evaluation.crossvalidation import calc_mean_std_scores
from model_evaluation.ranking_metrics import clef2019_average_precision
from pytorchic_bert.bert_embeddings import load_embedded_data
from text_classification.classifiers import embedding_classifier
from text_classification.classifiers.common import GenericClassifier
from text_classification.classifiers.embedding_classifier import EmbeddingClassifier
from text_classification.classifiers.tfidf_dataprocessor import regex_tokenizer
from text_classification.classifiers.tfidf_sgd_sklearn import TfIdfSGDSklearnClf


def rank_it(scores):
    return [r for r, score in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]

def benchmark(build_pipeline_fun,parameters,data):

    def score_fun(train_data, test_data):
        pipeline:GenericClassifier = build_pipeline_fun(**parameters)
        pipeline.fit(train_data)

        proba,y_train = pipeline.predict_proba_encode_targets(train_data)
        positive_index = pipeline.target_binarizer.classes_.tolist().index('1')
        ranking = rank_it(proba[:,positive_index].tolist())
        avgP_train = clef2019_average_precision(numpy.argmax(y_train,axis=1).tolist(), ranking)

        proba,y_test = pipeline.predict_proba_encode_targets(test_data)
        ranking = rank_it(proba[:,positive_index].tolist())
        avgP_test = clef2019_average_precision(numpy.argmax(y_test,axis=1).tolist(), ranking)

        return {
            'train-avgP': avgP_train,
            'test-avgP': avgP_test,
        }


    # splitter = ShuffleSplit(n_splits=5, test_size=0.2, random_state=111)
    splitter = GroupShuffleSplit(n_splits=4, test_size=0.2, random_state=111)
    splits = [(train, test) for train, test in
              splitter.split(X=range(len(data)), groups=[d['debatefile'] for d in data])]
    # test_file = '20190108_oval_office.tsv'
    # splits = [(
    #     np.array([i for i,d in enumerate(clef2019_data) if d['debatefile']!=test_file]),
    #     np.array([i for i,d in enumerate(clef2019_data) if d['debatefile']==test_file])
    #            )]
    m_scores_std_scores = calc_mean_std_scores(data, score_fun, splits)
    print(parameters)
    pprint(m_scores_std_scores['m_scores'])


def windowed_bow(data):
    raw_bows = [regex_tokenizer(d['utterance'])+['SPEAKER__'+d['speaker']] for d in data]

    def window_prefixed_bows(idx,before=-8,after=2):
        return [str(i)+'__'+tok
                for i in range(before,after)
                if idx+i<len(data) and idx+i>=0
                if data[idx+i]['debatefile']==data[idx]['debatefile']
                for tok in raw_bows[idx+i]]

    prefixed_bows = [window_prefixed_bows(idx) for idx in range(len(data))]
    return prefixed_bows


def raw_bow(data):
    raw_bows = [regex_tokenizer(d['utterance']) for d in data]
    return raw_bows

if __name__ == '__main__':
    from pathlib import Path
    home = str(Path.home())
    data = load_embedded_data(home+'/nlp/processed')
    # data = load_embedded_data('/tmp/processed')
    # data = get_Clef2019_data(home+'/code/misc/clef2019-factchecking-task1/data/training')
    pprint(Counter([d['label'] for d in data]))

    # benchmark(TfIdfSGDSklearnClf,{'process_data_to_bows_fun':windowed_bow,'alpha':0.00002,'get_targets_fun':lambda x:[x['label']]},data)
    # benchmark(TfIdfSGDSklearnClf,{'process_data_to_bows_fun':raw_bow,'alpha':0.00002,'get_targets_fun':lambda x:[x['label']]},data)
    benchmark(EmbeddingClassifier,parameters={'train_config':embedding_classifier.TrainConfig(n_epochs=19,lr=0.01)},data=data)

