import re
from time import time
from typing import List, Tuple

import sklearn_crfsuite
import spacy
import flair.datasets
from spacy.tokenizer import Tokenizer


class SpacyCrfSuiteTagger(object):

    def __init__(self,nlp = spacy.load('en_core_web_sm',disable=['parser'])
        ):
        self.nlp = nlp
        infix_re = re.compile(r'\s')
        self.nlp.tokenizer = Tokenizer(nlp.vocab,infix_finditer=infix_re.finditer)

    def fit(self,data:List[List[Tuple[str,str]]]):
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

if __name__ == '__main__':

    corpus = flair.datasets.UD_ENGLISH()
    print('train-data-len: %d'%len(corpus.train))
    print('test-data-len: %d'%len(corpus.test))

    tag_type = 'pos'
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)
    train_data = [[(token.text,token.tags['pos'].value) for token in datum] for datum in corpus.train]

    tagger = SpacyCrfSuiteTagger()
    tagger.fit(train_data)

    test_data = [[(token.text,token.tags['pos'].value) for token in datum] for datum in corpus.test]
    y_pred = tagger.predict(test_data)
    from sklearn_crfsuite import metrics

    targets = [[tag for token, tag in datum] for datum in test_data]

    labels = list(tagger.crf.classes_)

    metrics.flat_f1_score(targets, y_pred, average='weighted', labels=labels)

    # group B and I results
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(
        targets, y_pred, labels=sorted_labels, digits=3
    ))
