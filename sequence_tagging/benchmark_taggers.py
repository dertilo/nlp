import json
import logging
import multiprocessing
import sys
sys.path.append('.')

from functools import partial
from time import time

from sequence_tagging.flair_scierc_ner import TAG_TYPE, build_flair_sentences
from sequence_tagging.seq_tag_util import spanwise_pr_re_f1, bilou2bio
from sequence_tagging.spacy_features_sklearn_crfsuite import SpacyCrfSuiteTagger


from sklearn.model_selection import ShuffleSplit

from model_evaluation.crossvalidation import calc_mean_std_scores
from sequence_tagging.evaluate_flair_tagger import calc_train_test_spanwise_f1


import torch
torch.multiprocessing.set_start_method('spawn', force=True)
from pprint import pprint
from typing import List, Union

from commons import data_io
from flair.data import Sentence, Corpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from flair.training_utils import EvaluationMetric
from torch.utils.data import Dataset
from flair.models import SequenceTagger


def score_flair_tagger(
        splits,
        data:Union[List[Sentence],Dataset],

):
    from flair.trainers import ModelTrainer, trainer
    logger = trainer.log
    logger.setLevel(logging.WARNING)

    data_splits = [[data[i] for i in split] for split in splits]
    train_sentences,dev_sentences,test_sentences = data_splits

    corpus = Corpus(
        train=train_sentences,
        dev=dev_sentences,
        test=test_sentences, name='scierc')
    tag_dictionary = corpus.make_tag_dictionary(tag_type=TAG_TYPE)

    embedding_types: List[TokenEmbeddings] = [

        WordEmbeddings('glove'),

        # comment in this line to use character embeddings
        # CharacterEmbeddings(),

        # comment in these lines to use contextual string embeddings
        #
        # CharLMEmbeddings('news-forward'),
        #
        # CharLMEmbeddings('news-backward'),
    ]
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
    tagger: SequenceTagger = SequenceTagger(hidden_size=128,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=TAG_TYPE,
                                            locked_dropout=0.01,
                                            dropout=0.01,
                                            use_crf=True)
    trainer: ModelTrainer = ModelTrainer(tagger, corpus, optimizer=torch.optim.RMSprop)
    # print(tagger)
    # pprint([p_name for p_name, p in tagger.named_parameters()])
    save_path = 'flair_sequence_tagging/scierc-ner-%s'%multiprocessing.current_process()
    trainer.train('%s' % save_path, EvaluationMetric.MICRO_F1_SCORE,
                  learning_rate=0.01,
                  mini_batch_size=32,
                  max_epochs=2,
                  save_final_model=False
                  )
    # plotter = Plotter()
    # plotter.plot_training_curves('%s/loss.tsv' % save_path)
    # plotter.plot_weights('%s/weights.txt' % save_path)
    train_f1, test_f1 = calc_train_test_spanwise_f1(tagger, corpus.train, corpus.test, tag_name=TAG_TYPE)
    # calc_print_f1_scores(tagger,corpus.train,corpus.test,tag_name=TAG_TYPE)
    return {'f1-train':train_f1,'f1-test':test_f1}

def score_spacycrfsuite_tagger(splits,data,params={'c1':0.5,'c2':0.0}):
    data_splits = [[data[i] for i in split] for split in splits]
    train_data,dev_data,test_data = data_splits

    train_data = [[(token.text, token.tags['ner'].value) for token in datum] for datum in train_data]
    test_data = [[(token.text, token.tags['ner'].value) for token in datum] for datum in test_data]

    tagger = SpacyCrfSuiteTagger(**params)
    tagger.fit(train_data)

    y_pred = tagger.predict([[token for token, tag in datum] for datum in train_data])
    y_pred = [bilou2bio([tag for tag in datum]) for datum in y_pred]
    targets = [bilou2bio([tag for token, tag in datum]) for datum in train_data]
    _,_,f1_train = spanwise_pr_re_f1(y_pred, targets)

    y_pred = tagger.predict([[token for token, tag in datum] for datum in test_data])
    y_pred = [bilou2bio([tag for tag in datum]) for datum in y_pred]
    targets = [bilou2bio([tag for token, tag in datum]) for datum in test_data]
    _,_,f1_test = spanwise_pr_re_f1(y_pred, targets)
    return {'f1-train':f1_train,'f1-test':f1_test}


# def tune_hyperparams_spacycrfsuite():
#     def score_fun(data, **params):
#         splitter = ShuffleSplit(n_splits=3, test_size=0.2, random_state=111)
#         splits = [(train, train[:10], test) for train, test in splitter.split(X=range(len(data)))]
#         m_scores_std_scores = calc_mean_std_scores(lambda : data, partial(score_spacycrfsuite_tagger, params=params), splits)
#         print('%s; f1-score: %0.2f' % (json.dumps(params), m_scores_std_scores['m_scores']['f1-test']))
#         return m_scores_std_scores['m_scores']
#
#     from hyperopt import hp, Trials
#     import numpy as np
#     space = {
#         'c1': hp.loguniform('c1', np.log(10) * -6, 0),
#         'c2': hp.loguniform('c2', np.log(10) * -6, 0),
#     }
#     best, trials = tune_hyperparams(
#         data_supplier=get_data,
#         metric_for_hyperopt='f1-test',
#         max_evals=9,
#         score_fun=score_fun,
#         search_space=space,
#         trials=Trials()
#     )
#     for t in trials.results:
#         print('loss: %0.2f; f1-micro: %0.2f; params: %s' % (
#         t['loss'], t['mean-metrics']['f1-test'], json.dumps(t['hyperparams'])))
#     print(best)

def get_scierc_data_as_flair_sentences():
    # data_path = '/home/tilo/code/NLP/scisci_nlp/data/scierc_data/json/'
    data_path = '../data/scierc_data/json/'
    sentences = [sent for jsonl_file in ['train.json','dev.json','test.json']
                 for d in data_io.read_jsons_from_file('%s/%s' % (data_path,jsonl_file))
                 for sent in build_flair_sentences(d)]
    return sentences

if __name__ == '__main__':


    from json import encoder
    encoder.FLOAT_REPR = lambda o: format(o, '.2f')

    # tune_hyperparams_spacycrfsuite()
    sentences = get_scierc_data_as_flair_sentences()
    num_folds = 3
    splitter = ShuffleSplit(n_splits=num_folds, test_size=0.2, random_state=111)
    splits = [(train,train[:round(len(train)/5)],test) for train,test in splitter.split(X=range(len(sentences)))]
    # start = time()
    # m_scores_std_scores = calc_mean_std_scores(lambda : sentences, score_spacycrfsuite_tagger, splits)
    # print('spacy+crfsuite-tagger %d folds took: %0.2f seconds'%(num_folds,time()-start))
    # pprint(m_scores_std_scores)

    # start = time()
    # m_scores_std_scores = calc_mean_std_scores(get_data, score_spacycrfsuite_tagger, splits, n_jobs=min(multiprocessing.cpu_count() - 1, num_folds))
    # print('spacy+crfsuite-tagger %d folds-PARALLEL took: %0.2f seconds'%(num_folds,time()-start))
    # pprint(m_scores_std_scores)

    # FLAIR

    start = time()
    m_scores_std_scores = calc_mean_std_scores(get_scierc_data_as_flair_sentences, score_flair_tagger, splits)
    print('flair-tagger %d folds took: %0.2f seconds'%(num_folds,time()-start))
    pprint(m_scores_std_scores)

    start = time()
    m_scores_std_scores = calc_mean_std_scores(get_scierc_data_as_flair_sentences, score_flair_tagger, splits, n_jobs=3)
    print('flair-tagger %d folds PARALLEL took: %0.2f seconds'%(num_folds,time()-start))
    pprint(m_scores_std_scores)

