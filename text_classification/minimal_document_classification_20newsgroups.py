from collections import Counter
from pprint import pprint
from time import time

import numpy as np
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

def get_20newsgroups_data(train_test,
                          categories=None,
                          truncate_to:int=None,
                          min_num_tokens=0,
                          random_state=42):
    """
     'alt.atheism',
     'comp.graphics',
     'comp.os.ms-windows.misc',
     'comp.sys.ibm.pc.hardware',
     'comp.sys.mac.hardware',
     'comp.windows.x',
     'misc.forsale',
     'rec.autos',
     'rec.motorcycles',
     'rec.sport.baseball',
     'rec.sport.hockey',
     'sci.crypt',
     'sci.electronics',
     'sci.med',
     'sci.space',
     'soc.religion.christian',
     'talk.politics.guns',
     'talk.politics.mideast',
     'talk.politics.misc',
     'talk.religion.misc'
    """
    data = fetch_20newsgroups(subset=train_test,
                              shuffle=True,
                              remove=('headers', 'footers', 'quotes'),
                              categories=categories,
                              random_state=random_state
                              )
    target_names = data.target_names


    data = [(d, target_names[target]) for d, target in zip(data.data, data.target) if len(d.split(' ')) > min_num_tokens]
    if truncate_to is not None:
        def truncate(text):
            return text[0:min(len(text), truncate_to)]
        data = [(truncate(d),t) for d,t in data]
    return data


def encode_targets(data_train,data_test):
    train_labels = [label for _, label in data_train]
    test_labels = [label for _, label in data_test]
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)
    y_train = label_encoder.transform(train_labels)
    y_test = label_encoder.transform(test_labels)
    return label_encoder,y_train,y_test

def benchmark(clf):
    print('_' * 80)
    # print(clf)
    clf_descr = str(clf).split('(')[0]
    print(clf_descr)
    t0 = time()

    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred_train = clf.predict(X_train)
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_train, pred_train)
    print("train-f1-micro:   %0.3f" % score)
    score = metrics.accuracy_score(y_test, pred)
    print("test-f1-micro:   %0.3f" % score)

    return {'clf-name':clf_descr,
            'accuracy':np.round(score,2),
            'train-time':np.round(train_time,3),
            'test-time':np.round(test_time,3)}


if __name__ == '__main__':
    data_train = get_20newsgroups_data('train')
    data_test = get_20newsgroups_data('test')

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.75, min_df=2, max_features=30000,
                                 stop_words='english')
    X_train = vectorizer.fit_transform([text for text, _ in data_train])
    print("n_samples: %d, n_features: %d" % X_train.shape)

    X_test = vectorizer.transform([text for text, _ in data_test])
    print("n_samples: %d, n_features: %d" % X_test.shape)

    pprint(Counter([label for _, label in data_train]))

    label_encoder, y_train, y_test = encode_targets(data_train,data_test)

    benchmark(SGDClassifier(alpha=.00001, loss='log', penalty="elasticnet", l1_ratio=0.2))

    benchmark(MultinomialNB(alpha=.01))
