from typing import List

from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from flair.training_utils import EvaluationMetric
from flair.visual.training_curves import Plotter
import flair.datasets

corpus =  flair.datasets.UD_ENGLISH()
print(corpus)

tag_type = 'pos'

tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

embedding_types: List[TokenEmbeddings] = [WordEmbeddings('glove')]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=64,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

from flair.trainers import ModelTrainer
trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('resources/taggers/example-ner', EvaluationMetric.MICRO_F1_SCORE,
              learning_rate=0.1, mini_batch_size=32,
              max_epochs=2)

plotter = Plotter()
plotter.plot_training_curves('resources/taggers/example-ner/loss.tsv')
plotter.plot_weights('resources/taggers/example-ner/weights.txt')

'''
should reach 
MICRO_AVG: acc 0.6678 - f1-score 0.8008
MACRO_AVG: acc 0.4433 - f1-score 0.538004081632653
after 2 epochs on UD_ENGLISH
'''