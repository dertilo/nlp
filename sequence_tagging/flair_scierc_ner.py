import sys
sys.path.append('.')
from scipy.sparse import csr_matrix # TODO(tilo): if not imported before torch it throws: ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found (required by some-path-here/lib/python3.7/site-packages/scipy/sparse/_sparsetools.cpython-37m-x86_64-linux-gnu.so)

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


def read_scierc_data_to_FlairSentences(
    jsonl_file:str
    )->Dataset:
    def prefix_to_BIOES(label,start,end,current_index):
        if end - start > 0:
            if current_index == start:
                prefix = 'B'
            elif current_index == end:
                prefix = 'E'
            else:
                prefix = 'I'
        else:
            prefix = 'S'

        return prefix+'-'+label

    def tag_it(token:Token,index,ner_spans):
        labels = [(start,end,label) for start,end,label in ner_spans if index>=start and index<=end]

        if len(labels)>0:
            for start,end,label in labels:
                token.add_tag(TAG_TYPE, prefix_to_BIOES(label, start, end, index))
        else:
            token.add_tag(TAG_TYPE, 'O')

    def build_sentences(d:Dict)->List[Sentence]:
        offset=0
        sentences = []
        for tokens,ner_spans in zip(d['sentences'],d['ner']):
            sentence: Sentence = Sentence()
            [sentence.add_token(Token(tok)) for tok in tokens]
            [tag_it(token,k+offset,ner_spans) for k,token in enumerate(sentence)]
            offset+=len(tokens)
            sentences.append(sentence)

        return sentences

    dataset:Dataset = [sent for d in data_io.read_jsons_from_file(jsonl_file) for sent in build_sentences(d)]
    return dataset


if __name__ == '__main__':
    # 1. get the corpus
    data_path = '../data/scierc_data/json/'

    corpus = Corpus(
        train=read_scierc_data_to_FlairSentences('%strain.json' % data_path),
        dev=read_scierc_data_to_FlairSentences('%sdev.json' % data_path),
        test=read_scierc_data_to_FlairSentences('%stest.json' % data_path), name='scierc')
    pprint(Counter([tok.tags[TAG_TYPE].value for sent in corpus.train for tok in sent]))
    pprint(Counter([tok.tags[TAG_TYPE].value for sent in corpus.test for tok in sent]))

    # print(corpus)


    tag_dictionary = corpus.make_tag_dictionary(tag_type=TAG_TYPE)
    print(tag_dictionary.idx2item)

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

    # initialize sequence tagger
    from flair.models import SequenceTagger

    tagger: SequenceTagger = SequenceTagger(hidden_size=64,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=TAG_TYPE,
                                            locked_dropout=0.01,
                                            dropout=0.01,
                                            use_crf=True)

    # initialize trainer
    from flair.trainers import ModelTrainer

    trainer: ModelTrainer = ModelTrainer(tagger, corpus,optimizer=torch.optim.Adam)

    save_path = 'sequence_tagging/resources/taggers/scierc-ner'
    trainer.train('%s' % save_path, EvaluationMetric.MICRO_F1_SCORE,
                  learning_rate=0.01,
                  mini_batch_size=32,
                  max_epochs=1)

    # plotter = Plotter()
    # plotter.plot_training_curves('%s/loss.tsv' % save_path)
    # plotter.plot_weights('%s/weights.txt' % save_path)

    from sequence_tagging.evaluate_flair_tagger import evaluate_sequence_tagger, calc_print_f1_scores

    pprint('train-f1-macro: %0.2f'%evaluate_sequence_tagger(tagger,corpus.train)['f1-macro'])
    pprint('test-f1-macro: %0.2f'%evaluate_sequence_tagger(tagger,corpus.test)['f1-macro'])

    calc_print_f1_scores(tagger,corpus.train,corpus.test,tag_name=TAG_TYPE)