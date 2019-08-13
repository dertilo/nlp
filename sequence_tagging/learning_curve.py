import multiprocessing
import sys
from itertools import groupby
from typing import Dict, List, Tuple, Any, Iterable

sys.path.append('.')

from sklearn.model_selection import ShuffleSplit

from model_evaluation.crossvalidation import calc_scores, calc_mean_and_std

from pprint import pprint
from random import shuffle

from commons import data_io
import numpy as np

from sequence_tagging.benchmark_taggers import score_spacycrfsuite_tagger
from sequence_tagging.flair_scierc_ner import build_flair_sentences


def get_flair_sentences(file):
    return [seq for d in data_io.read_jsons_from_file(file) for seq in build_flair_sentences(d)]

len_train_set=None

def data_supplier():
    datasets = load_datasets()
    data = datasets['train'] + datasets['test']
    return data


def groupbykey(x:List[Dict]):
    return {k: [l for _, l in group]
            for k, group in
            groupby(sorted([(k, v) for d in x for k, v in d.items()], key=lambda x: x[0]), key=lambda x: x[0])
            }

def groupbyfirst(x:Iterable[Tuple[Any,Any]]):
    return {k: [l for _, l in group]
            for k, group in
            groupby(sorted([(k, v) for k,v in x], key=lambda x: x[0]), key=lambda x: x[0])
            }

data_path = '../data/scierc_data/json/'
results_path = '../data/scierc_data'

def load_datasets():
    # data_path = '/home/tilo/code/NLP/scisci_nlp/data/scierc_data/json/'
    datasets = {dataset_name: get_flair_sentences('%s/%s.json' % (data_path, dataset_name)) for dataset_name in
                ['train', 'dev', 'test']}
    return datasets

if __name__ == '__main__':
    data = load_datasets()
    len_train_and_test = len(data['train']+data['test'])
    len_train_set = len(data['train'])




    num_folds = 3
    splits=[(train_size,{'train': train , 'test':list(range(len_train_set,len_train_and_test))})
     for train_size in np.arange(0.1,1.0,0.1)
     for train,_ in ShuffleSplit(n_splits=num_folds, train_size=train_size,test_size=1-train_size, random_state=111).split(
                X=range(len_train_set))
     ]
    print('got %d evaluations to calculate'%len(splits))
    train_sizes = [t for t,_ in splits]
    scores = calc_scores(data_supplier,score_spacycrfsuite_tagger,[s for _,s in splits],n_jobs=min(multiprocessing.cpu_count(),len(splits)))
    results = groupbyfirst(zip(train_sizes,scores))

    def tuple_2_dict(t):
        m,s = t
        return {'mean':m,'std':s}
    
    data_io.write_json_to_file(results_path+'/learning_curve_scores.json',results)
    trainsize_to_mean_std_scores={train_size: tuple_2_dict(calc_mean_and_std(m))for train_size,m in results.items()}
    data_io.write_json_to_file(results_path+'/learning_curve_meanstd_scores.json',trainsize_to_mean_std_scores)
    # pprint(trainsize_to_mean_std_scores)
