import sys
sys.path.append('.')
from scipy.sparse import csr_matrix #TODO(tilo): if not imported before torch it throws: ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found

from sequence_tagging.seq_tag_util import spanwise_pr_re_f1

from pprint import pprint
import torch
from typing import List

from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from flair.training_utils import EvaluationMetric
from flair.visual.training_curves import Plotter
import flair.datasets

from sequence_tagging.evaluate_flair_tagger import calc_seqtag_eval_scores
from flair.models import SequenceTagger



corpus =  flair.datasets.UD_ENGLISH()
print(corpus)

tag_type = 'pos'

tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

embedding_types: List[TokenEmbeddings] = [WordEmbeddings('glove')]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)


tagger: SequenceTagger = SequenceTagger(hidden_size=32,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

from flair.trainers import ModelTrainer
trainer: ModelTrainer = ModelTrainer(tagger, corpus,optimizer=torch.optim.Adam)

trainer.train('resources/taggers/example-ner', EvaluationMetric.MICRO_F1_SCORE,
              learning_rate=0.01, mini_batch_size=32,
              max_epochs=3)

plotter = Plotter()
plotter.plot_training_curves('resources/taggers/example-ner/loss.tsv')
plotter.plot_weights('resources/taggers/example-ner/weights.txt')

train_sentences = corpus.train
test_sentences = corpus.test

train_data = [[(token.text, token.tags['pos'].value) for token in datum] for datum in train_sentences]
gold_targets_train = [[tag for token, tag in datum] for datum in train_data]
test_data = [[(token.text, token.tags['pos'].value) for token in datum] for datum in test_sentences]
gold_targets_test = [[tag for token, tag in datum] for datum in test_data]

pred_sentences = tagger.predict(train_sentences)
pred_data = [[token.tags['pos'].value for token in datum] for datum in pred_sentences]
pprint('train-f1-macro: %0.2f' % calc_seqtag_eval_scores(gold_targets_train, pred_data)['f1-macro'])

pred_sentences = tagger.predict(test_sentences)
pred_data = [[token.tags['pos'].value for token in datum] for datum in pred_sentences]
pprint('test-f1-macro: %0.2f' % calc_seqtag_eval_scores(gold_targets_test, pred_data)['f1-macro'])

#
# from sklearn_crfsuite import metrics
# print(metrics.flat_classification_report(
#     gold_targets, pred_test_data, labels=list(tagger.tag_dictionary.item2idx.keys()), digits=3
# ))

'''
on UD_ENGLISH it should reach: 
'train-f1-macro: 0.70'
'test-f1-macro: 0.69'
'''