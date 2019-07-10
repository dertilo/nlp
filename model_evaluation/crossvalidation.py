from typing import List
import numpy as np
from commons.util_methods import get_dict_paths, get_val, set_val
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit


def score_splits(scorer,data,splits):
    def calc_scores(split_indices):
        data_splits = [[data[i] for i in split] for split in split_indices]
        scoring = scorer(*data_splits)
        return scoring

    return [calc_scores(split) for split in splits]

def imputed(x):
    return np.NaN if isinstance(x,str) else x

def calc_mean_and_std(eval_metrices):
    assert isinstance(eval_metrices, list) and all([isinstance(d, dict) for d in eval_metrices])
    paths = []
    get_dict_paths(paths, [], eval_metrices[0])
    means = {}
    stds = {}
    for p in paths:
        try:
            m_val = np.mean([imputed(get_val(d,p)) for d in eval_metrices])
            set_val(means,p,m_val)
        except:
            print(p)
        try:
            std_val = np.std([imputed(get_val(d,p)) for d in eval_metrices])
            set_val(stds,p,std_val)
        except:
            print(p)

    return means,stds

    # if stratified:
    #     splitter = StratifiedShuffleSplit(n_splits, test_size, random_state=111)
    #     # y_categorical = np.argmax(targets_binarized, axis=1)
    #     splits = splitter.split(X=range(len(data)), y=[d['target'][0] for d in data])
    # else:
    #     splitter = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=111)
    #     splits = [(train,test) for train,test in splitter.split(X=range(len(data)))]

def calc_mean_std_scores(
        data:List,
        score_fun,
        splits,
    ):
    scores = score_splits(score_fun,data, splits) # TODO: this can fail if to few labels and splitter splitted unluckily
    m_scores, std_scores = calc_mean_and_std(scores)
    return {'m_scores':m_scores,
            'std_scores':std_scores}

