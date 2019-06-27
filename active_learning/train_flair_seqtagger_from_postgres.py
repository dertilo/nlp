import json
import sys
sys.path.append('.')
from scipy.sparse import csr_matrix # TODO(tilo): if not imported before torch it throws: ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found (required by some-path-here/lib/python3.7/site-packages/scipy/sparse/_sparsetools.cpython-37m-x86_64-linux-gnu.so)

from sqlalchemy_util.sqlalchemy_base import sqlalchemy_base, sqlalchemy_engine
from sqlalchemy import select, Table, Column, String
from sqlalchemy_util.sqlalchemy_methods import fetch_batch_wise_queueing

import torch
from collections import Counter
from pprint import pprint
from typing import List, Dict

from commons import data_io
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.data import Sentence, Token, Corpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings, CharacterEmbeddings
from flair.training_utils import EvaluationMetric
from flair.visual.training_curves import Plotter
from torch.utils.data import Dataset

TAG_TYPE = 'ner'


def prefix_to_BIOES(label, start, end, current_index):
    if end - start > 0:
        if current_index == start:
            prefix = 'B'
        elif current_index == end:
            prefix = 'E'
        else:
            prefix = 'I'
    else:
        prefix = 'S'

    return prefix + '-' + label


def tag_it(token: Token, index, ner_spans):
    labels = [(start, end, label) for start, end, label in ner_spans if index >= start and index <= end]

    if len(labels) > 0:
        for start, end, label in labels:
            token.add_tag(TAG_TYPE, prefix_to_BIOES(label, start, end, index))
    else:
        token.add_tag(TAG_TYPE, 'O')


def build_sentences(d: Dict, annotator_names=[]) -> List[Sentence]:

    sentences = [build_flair_Sentence(tokens) for tokens in d['sentences']]

    for annotator in annotator_names:
        if annotator in d['ner']:
            offset = 0
            for sentence, ner_spans in zip(sentences, d['ner'][annotator]):
                [tag_it(token, k + offset, ner_spans) for k, token in enumerate(sentence)]
                offset += len(sentence)
    return sentences

def build_flair_Sentence(tokens):
    sentence: Sentence = Sentence()
    [sentence.add_token(Token(tok)) for tok in tokens]
    return sentence

def read_scierc_data_to_FlairSentences(table:Table)->Dataset:
    colnames = [c.name for c in table.columns]
    assert all([c in colnames for c in ['sentences','ner','relations','clusters']])

    q = select([table]).limit(1000000)
    def process_row(d):
        datum = {c.name:json.loads(d[c.name]) for c in table.columns}
        return datum
    data_g = (process_row(d) for d in fetch_batch_wise_queueing(q, sqlalchemy_engine, batch_size=100))
    dataset:Dataset = [sent for d in data_g for sent in build_sentences(d)]
    return dataset


def train_seqtagger(train_data:Dataset,
                    dev_data:Dataset,
                    test_data:Dataset
                    ):
    corpus = Corpus(
        train=train_data,
        dev=dev_data,
        test=test_data,
        name='scierc')

    pprint(Counter([tok.tags[TAG_TYPE].value for sent in corpus.train for tok in sent]))

    tag_dictionary = corpus.make_tag_dictionary(tag_type=TAG_TYPE)
    print(tag_dictionary.idx2item)

    embedding_types: List[TokenEmbeddings] = [ WordEmbeddings('glove')]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    from flair.models import SequenceTagger

    tagger: SequenceTagger = SequenceTagger(hidden_size=64,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=TAG_TYPE,
                                            locked_dropout=0.01,
                                            dropout=0.01,
                                            use_crf=True)

    from flair.trainers import ModelTrainer

    trainer: ModelTrainer = ModelTrainer(tagger, corpus,optimizer=torch.optim.Adam)

    save_path = 'sequence_tagging/resources/taggers/scierc-ner'
    trainer.train('%s' % save_path, EvaluationMetric.MICRO_F1_SCORE,
                  learning_rate=0.01,
                  mini_batch_size=32,
                  max_epochs=20)

    # plotter = Plotter()
    # plotter.plot_training_curves('%s/loss.tsv' % save_path)
    # plotter.plot_weights('%s/weights.txt' % save_path)

    from sequence_tagging.evaluate_flair_tagger import evaluate_sequence_tagger
    pprint('train-f1-macro: %0.2f'%evaluate_sequence_tagger(tagger,corpus.train)['f1-macro'])
    pprint('dev-f1-macro: %0.2f'%evaluate_sequence_tagger(tagger,corpus.dev)['f1-macro'])
    pprint('test-f1-macro: %0.2f'%evaluate_sequence_tagger(tagger,corpus.test)['f1-macro'])
    return tagger

if __name__ == '__main__':
    table_name = 'scierc'
    columns = [Column('id', String, primary_key=True)] + [Column(colname, String) for colname in ['sentences','ner','relations','clusters']]
    table = Table(table_name, sqlalchemy_base.metadata, *columns, extend_existing=True)

    train_data = read_scierc_data_to_FlairSentences(table)
    train_seqtagger(train_data,train_data,train_data)