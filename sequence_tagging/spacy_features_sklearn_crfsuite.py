import re
from collections import Counter
from pprint import pprint
from time import time
from typing import List, Tuple

import sklearn_crfsuite
import spacy
import flair.datasets
from flair.data import Dictionary
from spacy.tokenizer import Tokenizer

from sequence_tagging.evaluate_flair_tagger import calc_seqtag_eval_scores


class SpacyCrfSuiteTagger(object):

    def __init__(self,nlp = spacy.load('en_core_web_sm',disable=['parser'])
        ):
        self.nlp = nlp
        infix_re = re.compile(r'\s')
        self.nlp.tokenizer = Tokenizer(nlp.vocab,infix_finditer=infix_re.finditer)

    def fit(self,data:List[List[Tuple[str,str]]]):

        # tag_counter = Counter([tag for sent in data for _,tag in sent])
        # tag2count = {t: c for t, c in tag_counter.items() if t != 'O'}
        # # print(tag2count)
        #
        # dictionary = Dictionary()
        # [dictionary.add_item(t) for t in tag2count]
        # dictionary.add_item('O')

        start = time()
        processed_data = [self.extract_features_with_spacy(datum) for datum in data]
        print('spacy-processing train-data took: %0.2f'%(time()-start))

        self.crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=20, all_possible_transitions=True)
        targets = [[tag for token, tag in datum] for datum in data]
        start = time()
        self.crf.fit(processed_data, targets)
        print('crfsuite-fitting took: %0.2f'%(time()-start))

    def extract_features_with_spacy(self, datum):
        text = ' '.join([token for token, _ in datum])
        doc = self.nlp(text)
        assert len(doc) == len(datum)
        return [
            {'text':token.text,'lemma':token.lemma_,'pos':token.pos_,
             # 'dep':token.dep_,
             'shape':token.shape_,'is_alpha':token.is_alpha,'is_stop':token.is_stop}
            for token in doc
        ]

    def predict(self,data):
        processed_data = [self.extract_features_with_spacy(datum) for datum in data]
        y_pred = self.crf.predict(processed_data)
        return y_pred


def get_UD_English_data():

    corpus = flair.datasets.UD_ENGLISH()
    train_data_flair = corpus.train
    test_data_flair = corpus.test
    print('train-data-len: %d' % len(train_data_flair))
    print('test-data-len: %d' % len(test_data_flair))

    tag_type = 'pos'

    def filter_tags(tag):
        return tag# if tag_counter[tag] > 50 else 'O'

    train_data = [[(token.text, filter_tags(token.tags['pos'].value)) for token in datum] for datum in train_data_flair]
    test_data = [[(token.text, filter_tags(token.tags['pos'].value)) for token in datum] for datum in test_data_flair]
    return train_data, test_data,tag_type

if __name__ == '__main__':

    train_data, test_data,tag_type = get_UD_English_data()

    tagger = SpacyCrfSuiteTagger()
    tagger.fit(train_data)

    y_pred = tagger.predict(test_data)
    from sklearn_crfsuite import metrics

    targets = [[tag for token, tag in datum] for datum in test_data]

    # labels = list(tagger.crf.classes_)
    #
    # metrics.flat_f1_score(targets, y_pred, average='weighted', labels=labels)
    #
    # # group B and I results
    # sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    #
    # print(metrics.flat_classification_report(
    #     targets, y_pred, labels=sorted_labels, digits=3
    # ))

    pprint('test-f1-macro: %0.2f' % calc_seqtag_eval_scores(targets, y_pred)['f1-macro'])

'''
spacy-processing train-data took: 66.69
crfsuite-fitting took: 31.05
    'test-f1-macro: 0.70'
'''