from pprint import pprint
from typing import List

import numpy as np
from flair.data import TaggedCorpus, Sentence
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from sklearn import metrics
import torch.nn.functional as F
from sklearn.preprocessing import MultiLabelBinarizer

from model_evaluation.classification_metrics import calc_classification_metrics
from text_classification.data_readers import get_20newsgroups_data

def build_Sentences(text_label_tuples):
    sentences = [Sentence(text, labels=[label], use_tokenizer=True) for text,label in text_label_tuples]
    return [s for s in sentences if len(s.tokens) > 0]

class TextClassifierProba(TextClassifier):

    def predict_proba(self,data):
        probas = F.softmax(self.forward(data), dim=1)
        return probas.detach().numpy().astype('float64')


def get_targets(sentences:List[Sentence]):
    return [[l.value for l in s.labels] for s in sentences]


if __name__ == '__main__':
    sentences_train = build_Sentences(get_20newsgroups_data('train', min_num_tokens=5, truncate_to=200))
    sentences_dev = build_Sentences(get_20newsgroups_data('train', min_num_tokens=5, truncate_to=200))
    sentences_test = build_Sentences(get_20newsgroups_data('test', min_num_tokens=5, truncate_to=200))

    corpus: TaggedCorpus = TaggedCorpus(sentences_train, sentences_dev, sentences_test)
    label_dict = corpus.make_label_dictionary()
    word_embeddings = [WordEmbeddings('glove'),

                       # comment in flair embeddings for state-of-the-art results
                       # FlairEmbeddings('news-forward'),
                       # FlairEmbeddings('news-backward'),
                       ]

    document_embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(word_embeddings,
                                                                         hidden_size=32,
                                                                         reproject_words=True,
                                                                         reproject_words_dimension=word_embeddings[0].embedding_length,
                                                                         )




    label_encoder = MultiLabelBinarizer()
    label_encoder.fit(get_targets(sentences_train))

    def score_fun(train_data,test_data):
        clf = TextClassifierProba(document_embeddings, label_dictionary=label_dict, multi_label=False)
        trainer = ModelTrainer(clf, corpus)
        base_path = 'flair_resources/taggers/ag_news'
        print('start training')
        trainer.train(base_path,
                      learning_rate=0.1,
                      mini_batch_size=32,
                      anneal_factor=0.5,
                      patience=5,
                      max_epochs=10)

        y_train = label_encoder.transform(get_targets(train_data))
        y_test = label_encoder.transform(get_targets(test_data))

        proba = clf.predict_proba(train_data)
        pred = np.array(proba == np.expand_dims(np.max(proba, axis=1), 1),dtype='int64')
        target_names = label_encoder.classes_.tolist()
        train_scores = calc_classification_metrics(proba, pred, y_train, target_names=target_names)

        proba = clf.predict_proba(test_data)
        pred = np.array(proba == np.expand_dims(np.max(proba, axis=1), 1),dtype='int64')

        test_scores = calc_classification_metrics(proba, pred, y_test, target_names=target_names)

        return {
            'train':train_scores,
            'test':test_scores,
        }


    scores = score_fun(sentences_train, sentences_test)
    pprint(scores['train']['f1-micro'])
    pprint(scores['test']['f1-micro'])
