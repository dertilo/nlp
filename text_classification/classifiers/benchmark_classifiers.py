import sys

from text_classification.classifiers import embedding_classifier
from text_classification.classifiers.embedding_classifier import EmbeddingClassifier

sys.path.append('.')
from functools import partial
from scipy.sparse import csr_matrix

import numpy

from pytorchic_bert.bert_embeddings import load_embedded_data

from pytorchic_bert.selfattention_encoder import BertConfig
from pytorchic_bert.utils import set_seeds

from getting_data.sentiment140 import get_sentiment140_data
from text_classification.classifiers.attention_based_classifier import AttentionClassifierPytorch, TrainConfig
from text_classification.classifiers.embedding_bag_pytorch import EmbeddingBagClassifier
from text_classification.classifiers.tfidf_dataprocessor import raw_bow
from text_classification.classifiers.tfidf_elasticnet_pytorch import TfIdfElasticNetPytorchClf

from collections import Counter
from pprint import pprint
from typing import Dict

from sklearn.model_selection import ShuffleSplit
import numpy as np

from model_evaluation.classification_metrics import calc_classification_metrics
from model_evaluation.crossvalidation import calc_mean_std_scores
from text_classification.classifiers.common import GenericClassifier
from text_classification.classifiers.tfidf_sgd_sklearn import TfIdfSGDSklearnClf
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


    splitter = ShuffleSplit(n_splits=3, test_size=0.2, random_state=111)
    splits = [(train, test) for train, test in splitter.split(X=range(len(data)))]

    m_scores_std_scores = calc_mean_std_scores(data, score_fun, splits)
    print(build_pipeline_fun.__name__,parameters)
    pprint(m_scores_std_scores['m_scores'])

if __name__ == '__main__':
    from pathlib import Path
    home = str(Path.home())
    # max_text_len = 400
    # data = [{'text':text,'labels':[label]} for text,label in get_20newsgroups_data('train', max_text_len=500)]
    # data = [{'text':d['text'][:min(len(d['text']),max_text_len)],'labels':[str(d['polarity'])]} for d in get_sentiment140_data(datafile=home+'/data/Sentiment140/training.1600000.processed.noemoticon.csv')]
    # data = [data[i] for i in numpy.random.random_integers(0,len(data)-1,size=(100_000,))]
    data = load_embedded_data('/tmp/processed')
    label_counter = Counter([d['labels'][0] for d in data])
    pprint(label_counter)

    benchmark(TfIdfSGDSklearnClf, parameters={'text_to_bow_fun':raw_bow, 'alpha':0.0001}, data=data)
    # benchmark(TfIdfElasticNetPytorchClf,parameters={'text_to_bow_fun':raw_bow,'alpha':0.00002},data=data)
    # benchmark(EmbeddingBagClassifier,parameters={'embedding_dim':3,'alpha':0.0},data=data)
    benchmark(EmbeddingClassifier,parameters={'train_config':embedding_classifier.TrainConfig(n_epochs=19,lr=0.01)},data=data)
    #
    # cfg = TrainConfig(seed=2,n_epochs=20,lr=0.001,patience=3,tol=0.0)#.from_json('pytorchic_bert/config/train_mrpc.json')
    # # model_cfg = BertConfig(
    # #     n_heads=8,
    # #     vocab_size=30522,#TODO(tilo):WTF!
    # #     dim=64,dim_ff=4*64,n_layers=2)
    # model_cfg = BertConfig.from_json('pytorchic_bert/config/bert_base.json')
    #
    # max_len = 128
    # set_seeds(cfg.seed)
    #
    # vocab = home+'/data/models/uncased_L-12_H-768_A-12/vocab.txt'
    # dp = TwoSentDataProcessor(vocab_file=vocab, max_len=max_len)
    #
    # # model_cfg = 'pytorchic_bert/config/bert_tiny.json'
    # # import pytorchic_bert.selfattention_encoder as selfatt_enc
    # # cfg = AttentionClassifierConfig(lr=1e-4, n_epochs=1, batch_size=128)
    # # model_cfg = selfatt_enc.BertConfig.from_json(model_cfg)
    # # max_len = model_cfg.max_len
    # #
    # # vocab_file = home + '/data/models/uncased_L-12_H-768_A-12/vocab.txt'
    # #
    # parameters={'train_config':cfg, 'model_config':model_cfg, 'dataprocessor':dp,
    #             'pretrain_file': home + '/data/models/uncased_L-12_H-768_A-12/bert_encoder_pytorch.pt'
    #             }
    # [d.__setitem__('textb', '') for d in data]
    #
    # benchmark(AttentionClassifierPytorch, parameters=parameters, data=data)
