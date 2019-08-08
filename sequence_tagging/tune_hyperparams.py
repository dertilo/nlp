import multiprocessing
import sys
from time import time

sys.path.append('.')

import json
from functools import partial

from commons import data_io
from ray.tune.suggest.hyperopt import HyperOptSearch
from sklearn.model_selection import ShuffleSplit

from ray import tune
from ray.tune import track, Trainable

from model_evaluation.crossvalidation import calc_mean_std_scores
from sequence_tagging.benchmark_taggers import score_spacycrfsuite_tagger
from sequence_tagging.flair_scierc_ner import build_flair_sentences

import os
base_path = os.path.dirname(os.path.abspath(__file__))


def get_data():
    # data_path = '/home/tilo/code/NLP/scisci_nlp/data/scierc_data/json'
    data_path = '/docker-share/data/scierc_data/json'
    # data_path = '/home/users/t/tilo-himmelsbach/data/scierc_data/json'
    sentences = [sent for jsonl_file in ['train.json', 'dev.json', 'test.json']
                 for d in data_io.read_jsons_from_file('%s/%s' % (data_path, jsonl_file))
                 for sent in build_flair_sentences(d)]
    return sentences

data = None
def crosseval_configuration(config):
    global data
    if data is None:
        data = get_data()

    splitter = ShuffleSplit(n_splits=3, test_size=0.2, random_state=111)
    splits = [(train, train[:10], test) for train, test in splitter.split(X=range(len(data)))]
    m_scores_std_scores = calc_mean_std_scores(lambda: data, partial(score_spacycrfsuite_tagger, params=config), splits)
    track.log(f1=m_scores_std_scores['m_scores']['f1-test'])


if __name__ == "__main__":
    from hyperopt import hp
    import numpy as np

    space = {
        'c1': hp.loguniform('c1', np.log(10) * -6, 0),
        'c2': hp.loguniform('c2', np.log(10) * -6, 0),
    }

    algo = HyperOptSearch(space, max_concurrent=multiprocessing.cpu_count()-1,metric='f1')
    start = time()
    analysis = tune.run(
        crosseval_configuration,
        search_alg=algo,
        name="exp",
        stop={
            "training_iteration": 1
        },
        num_samples=9,
        local_dir=base_path,
        reuse_actors=False
    )
    print('hyperparam-tuning took: %0.2f seconds'%(time()-start))
    print("Best config is:", analysis.get_best_config(metric="f1"))

    '''
Result for crosseval_stuff_5_c1=0.0040206,c2=0.04865:
f1: 0.5349294151738395

hyperparam-tuning took: 492.19 seconds
Best config is: {'c1': 0.004020607666635166, 'c2': 0.048649565875974}

    '''