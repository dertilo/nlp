import logging
import multiprocessing
import sys
sys.path.append('.')

from time import time

from sequence_tagging.flair_scierc_ner import TAG_TYPE, build_flair_sentences
from sequence_tagging.seq_tag_util import spanwise_pr_re_f1, bilou2bio, calc_seqtag_f1_scores
from sequence_tagging.spacy_features_sklearn_crfsuite import SpacyCrfSuiteTagger
from sklearn.model_selection import ShuffleSplit

from model_evaluation.crossvalidation import calc_mean_std_scores

import torch
torch.multiprocessing.set_start_method('spawn', force=True)
from pprint import pprint
from typing import List, Union, Dict

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

    data_splits = {split_name:[data[i] for i in split] for split_name,split in splits.items()}

    train_sentences,dev_sentences,test_sentences = data_splits['train'],data_splits['dev'],data_splits['test'],

    corpus = Corpus(
        train=train_sentences,
        dev=dev_sentences,
        test=test_sentences, name='scierc')
    tag_dictionary = corpus.make_tag_dictionary(tag_type=TAG_TYPE)

    embedding_types: List[TokenEmbeddings] = [
        WordEmbeddings('glove'),
    ]
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
    tagger: SequenceTagger = SequenceTagger(hidden_size=64,
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
                  max_epochs=19,
                  patience=3,
                  save_final_model=False
                  )
    # plotter = Plotter()
    # plotter.plot_training_curves('%s/loss.tsv' % save_path)
    # plotter.plot_weights('%s/weights.txt' % save_path)

    def flair_tagger_predict_bio(sentences: List[Sentence]):
        train_data = [[(token.text, token.tags[tagger.tag_type].value) for token in datum] for datum in sentences]
        targets = [bilou2bio([tag for token, tag in datum]) for datum in train_data]

        pred_sentences = tagger.predict(sentences)
        pred_data = [bilou2bio([token.tags[tagger.tag_type].value for token in datum]) for datum in pred_sentences]


        return pred_data,targets

    return {
        'train':calc_seqtag_f1_scores(flair_tagger_predict_bio,corpus.train),
        'test':calc_seqtag_f1_scores(flair_tagger_predict_bio,corpus.test)
    }

def score_spacycrfsuite_tagger(splits:Dict[str,List[int]],data,params={'c1':0.5,'c2':0.0}):
    data_splits = {split_name:[data[i] for i in split] for split_name,split in splits.items()}

    def get_data_of_split(split_name):
        return [[(token.text, token.tags['ner'].value) for token in datum] for datum in data_splits[split_name]]

    tagger = SpacyCrfSuiteTagger(**params)
    tagger.fit(get_data_of_split('train'))

    def pred_fun(token_tag_sequences):
        y_pred = tagger.predict([[token for token, tag in datum] for datum in token_tag_sequences])
        y_pred = [bilou2bio([tag for tag in datum]) for datum in y_pred]
        targets = [bilou2bio([tag for token, tag in datum]) for datum in token_tag_sequences]
        return y_pred,targets

    return {split_name: calc_seqtag_f1_scores(pred_fun,get_data_of_split(split_name)) for split_name in data_splits.keys()}



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

    sentences = get_scierc_data_as_flair_sentences()
    num_folds = 3
    splitter = ShuffleSplit(n_splits=num_folds, test_size=0.2, random_state=111)
    splits = [{'train':train,'dev':train[:round(len(train)/5)],'test':test} for train,test in splitter.split(X=range(len(sentences)))]

    # spacy + crf-suite

    start = time()
    m_scores_std_scores = calc_mean_std_scores(get_scierc_data_as_flair_sentences, score_spacycrfsuite_tagger, splits, n_jobs=min(multiprocessing.cpu_count() - 1, num_folds))
    print('spacy+crfsuite-tagger %d folds-PARALLEL took: %0.2f seconds'%(num_folds,time()-start))
    pprint(m_scores_std_scores)

    # FLAIR

    start = time()
    n_jobs = min(5, num_folds)
    m_scores_std_scores = calc_mean_std_scores(get_scierc_data_as_flair_sentences, score_flair_tagger, splits, n_jobs=n_jobs)
    print('flair-tagger %d folds with %d jobs in PARALLEL took: %0.2f seconds'%(num_folds,n_jobs,time()-start))
    pprint(m_scores_std_scores)


'''
#############################################################################
on gunther

spacy+crfsuite-tagger 3 folds-PARALLEL took: 122.88 seconds
{'m_scores': {'dev': {'f1-macro': 0.8822625032484681,
                      'f1-micro': 0.9528343173272004,
                      'f1-spanwise': 0.8470436086284675},
              'test': {'f1-macro': 0.5742946309433821,
                       'f1-micro': 0.832899550463387,
                       'f1-spanwise': 0.5345123493111902},
              'train': {'f1-macro': 0.8844589822247658,
                        'f1-micro': 0.9522832740014087,
                        'f1-spanwise': 0.842115934181045}},
 'std_scores': {'dev': {'f1-macro': 0.009338633769168965,
                        'f1-micro': 0.0020278574245488883,
                        'f1-spanwise': 0.007549419792021609},
                'test': {'f1-macro': 0.019814579353249383,
                         'f1-micro': 0.003716883368130915,
                         'f1-spanwise': 0.00622111159374358},
                'train': {'f1-macro': 0.002357299459853917,
                          'f1-micro': 0.0006117207995410837,
                          'f1-spanwise': 0.0025053074858217297}}}

flair-tagger 3 folds with 3 jobs in PARALLEL took: 1076.74 seconds
{'m_scores': {'test': {'f1-macro': 0.5293097909510559,
                       'f1-micro': 0.810281415662149,
                       'f1-spanwise': 0.49897266786217936},
              'train': {'f1-macro': 0.7423535896182377,
                        'f1-micro': 0.8933529713434439,
                        'f1-spanwise': 0.7057908851327634}},
 'std_scores': {'test': {'f1-macro': 0.0221113842260729,
                         'f1-micro': 0.02028681159209277,
                         'f1-spanwise': 0.03217935301111879},
                'train': {'f1-macro': 0.024130587949076823,
                          'f1-micro': 0.016407352682141038,
                          'f1-spanwise': 0.02149275589157383}}}
'''