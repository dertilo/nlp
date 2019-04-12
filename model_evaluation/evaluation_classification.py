import abc
import os
from functools import partial
from multiprocessing.pool import Pool
from time import time
from typing import Tuple, List, Iterable, Callable

import numpy as np
import pandas
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from commons_old.util_methods import get_dict_paths, set_val, get_val


# def average_precision_score(y_true, y_score, average="macro",
#                             sample_weight=None):
#
#     def _binary_uninterpolated_average_precision(
#             y_true, y_score, sample_weight=None):
#         precision, recall, thresholds = precision_recall_curve(
#             y_true, y_score, sample_weight=sample_weight)
#         # Return the step function integral
#         # The following works because the last entry of precision is
#         # guaranteed to be 1, as returned by precision_recall_curve
#         of_interest = [k for k in range(len(precision)) if precision[k]>0.5]
#         precision = [precision[k] for k in of_interest]
#         recall = [recall[k] for k in of_interest]
#         return -np.sum(np.diff(recall) * np.array(precision)[:-1])
#
#     return _average_binary_score(_binary_uninterpolated_average_precision,
#                                  y_true, y_score, average,
#                                  sample_weight=sample_weight)


class ToBeScoredAbstractClassifier(object):
    def __init__(self):
        pass
    @abc.abstractmethod
    def _predict(self,data:List)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
        raise NotImplementedError()

    def predict(self,data:Iterable)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
        prediction,pred_probas, target_binarized = self._predict(data)
        assert isinstance(pred_probas,np.ndarray)
        pred_probas = pred_probas.astype('float64')
        prediction = prediction.astype('float64')
        assert isinstance(prediction,np.ndarray)
        assert isinstance(target_binarized,np.ndarray) and target_binarized.dtype == 'int64'
        assert prediction.shape==target_binarized.shape==pred_probas.shape
        return prediction,pred_probas, target_binarized

    @abc.abstractmethod
    def fit(self,train_data):
        raise NotImplementedError()

def try_to_calculate(fun,x,y,default=None):
    try:
        return fun(x,y)
    except:
        return default

def calc_interpol_pr_auc(target,pred):
    pre, rec, threshs = metrics.precision_recall_curve(target,pred, pos_label=1)
    pr_auc = metrics.auc(pre, rec, reorder=True)
    return pr_auc


def calc_pr_auc_interpolated_labelwise(y_target,y_pred_proba,target_names):
    return [try_to_calculate(calc_interpol_pr_auc,y_target[:, target_names.index(target)], y_pred_proba[:, target_names.index(target)],-1) for target in target_names]


def classification_scores(y_pred_proba, y_pred, y_target, target_names,time_it=False):
    if time_it: start = time()
    assert isinstance(y_target, np.ndarray) and y_target.dtype == 'int64'
    assert isinstance(y_pred_proba, np.ndarray) and y_pred_proba.dtype == 'float64'
    assert isinstance(y_pred, np.ndarray) and y_pred.dtype == 'float64'

    report_txt = metrics.classification_report(y_true=y_target,
                                               y_pred=y_pred,
                                               target_names=target_names, digits=3)

    pr_auc_labelwise = try_to_calculate(partial(average_precision_score,average=None),y_target,y_pred_proba,default=[-1 for _ in range(len(target_names))])
    pr_auc_interpolated_labelwise = calc_pr_auc_interpolated_labelwise(y_target,y_pred_proba,target_names)
    roc_auc_labelwise = try_to_calculate(partial(metrics.roc_auc_score,average=None),y_target,y_pred_proba,default=[-1 for _ in range(len(target_names))])
    auc_macro = try_to_calculate(partial(metrics.roc_auc_score,average='macro'),y_target,y_pred_proba)
    pr_auc_macro = try_to_calculate(partial(average_precision_score,average='macro'),y_target,y_pred_proba)
    auc_micro = try_to_calculate(partial(metrics.roc_auc_score,average='micro'),y_target,y_pred_proba)
    pr_auc_micro = try_to_calculate(partial(average_precision_score,average='micro'),y_target,y_pred_proba)
    # labelwise_pr_curve = {target_names[k]:precision_recall_curve(y_target[:,k],y_pred_proba[:,k]) for k in range(len(target_names))}
    metrices = {
        'labelwise': classifaction_report_to_dict(report_txt),
        'ROC-AUC-macro': auc_macro,
        'PR-AUC-macro': pr_auc_macro,
        'ROC-AUC-micro': auc_micro,
        'PR-AUC-micro': pr_auc_micro,
        'f1-macro': metrics.f1_score(y_target, y_pred, average='macro'),
        'f1-micro': metrics.f1_score(y_target, y_pred, average='micro'),
        # 'accuracy': metrics.accuracy_score(y_target,y_pred) # same as f1-micro for single-label classification
    }
    for label, roc_auc,pr_auc,pr_auc_interp in zip(target_names,roc_auc_labelwise,pr_auc_labelwise,pr_auc_interpolated_labelwise):
        metrices['labelwise'][label]['ROC-AUC']=0.001*round(1000*roc_auc)
        metrices['labelwise'][label]['PR-AUC']=0.001*round(1000*pr_auc)
        metrices['labelwise'][label]['PR-AUC-interp']=0.001*round(1000*pr_auc_interp)
        # metrices['labelwise'][label]['pr_curve']=labelwise_pr_curve[label]
    if time_it: print('calculating classification scores took: %0.2f seconds'%(time()-start))
    return metrices


def multilabel_scorer(toBeScoredWrapper:ToBeScoredAbstractClassifier, train_data, test_data, target_names,
                      train_it=True,time_it=False
                      ):
    assert isinstance(target_names,list) and all([isinstance(s,str) for s in target_names])
    if train_it:
        toBeScoredWrapper.fit(train_data)
    pred,pred_proba,targets = toBeScoredWrapper.predict(train_data)
    train_scores = classification_scores(pred_proba, pred, targets, target_names,time_it)
    pred,pred_proba,targets = toBeScoredWrapper.predict(test_data)
    test_scores = classification_scores(pred_proba, pred, targets, target_names,time_it)

    # if threshtuner is not None:
    #     threshtuner.collect(pred_proba,targets)
    return {'train': train_scores,
            'test': test_scores,
            }


evaluator_job=None
class EvaluatorJob(object):
    def __init__(self,model,scorer,data):
        self.model = model
        self.scorer = scorer
        self.data = data

    def train_and_score(self,train_test_idx):
        train_idx = train_test_idx[0]
        test_idx = train_test_idx[1]

        train_data = [self.data[i] for i in train_idx]
        test_data = [self.data[i] for i in test_idx]
        scoring = self.scorer(self.model,train_data,test_data)
        # if keras.backend.backend() == 'tensorflow':
        #     keras.backend.clear_session()
        return scoring
def init_fun(model_supplier, scorer, data):
    global evaluator_job
    evaluator_job = EvaluatorJob(model_supplier(), scorer, data)
def fun(arg):
    return evaluator_job.train_and_score(arg)

def crosseval(model_supplier, splits, data, scorer,n_jobs=1):
    args = [(train_idx,test_idx) for train_idx, test_idx in splits]
    if n_jobs>1:
        assert False
        pool = Pool(processes=n_jobs, initializer=init_fun, initargs=(model_supplier,scorer,data))
        scores = pool.map(fun,args)
        pool.close()
    else:
        scores = []
        c = 0
        for train_idx,test_idx in args:
            c+=1
            train_data = [data[i] for i in train_idx]
            test_data = [data[i] for i in test_idx]
            scoring = scorer(model_supplier(), train_data, test_data)
            scores.append(scoring)

    assert len(scores)>0
    return scores

def crosseval_mean_std_scores(model_supplier, splits,data, scorer,n_jobs=1):
    # try:
    scores = crosseval(model_supplier, splits, data, scorer,n_jobs) # TODO: this can fail if to few labels and splitter splitted unluckily
    return calc_mean_and_std(scores)
    # except Exception:
    #     return {}

def imputed(x):
    return np.NaN if isinstance(x,str) else x
# from matplotlib import pyplot as plt

def calc_mean_and_std(eval_metrices):
    assert isinstance(eval_metrices, list) and all([isinstance(d, dict) for d in eval_metrices])
    paths = []
    get_dict_paths(paths, [], eval_metrices[0])
    means = {}
    stds = {}
    for p in paths:
        if 'pr_curve' in p[-1]:
            assert False
            # pr_curves = [get_val(d,p) for d in eval_metrices]
            # plt.figure()
            # plt.title('-'.join(p))
            # [plt.plot(rec, pre) for pre, rec, _ in pr_curves]

        else:
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
    # plt.show()
    return means,stds


def scores_to_csv(scores,dump_file='/tmp/scores.csv'):
    labels = [l for l in scores['train']['labelwise'].keys()]
    get_values = lambda scores, traintest, name: [scores[traintest]['labelwise'][l][name] for l in labels]
    df = pandas.DataFrame(data={
        'f1-scores-test': get_values(scores, 'test', 'f1_score'),
        'f1-scores-train': get_values(scores, 'train', 'f1_score'),
        'precision-test': get_values(scores, 'test', 'precision'),
        'precision-train': get_values(scores, 'train', 'precision'),
        'recall-test': get_values(scores, 'test', 'recall'),
        'recall-train': get_values(scores, 'train', 'recall'),
        'support-test': get_values(scores, 'test', 'support'),
        'support-train': get_values(scores, 'train', 'support'),

    },
        index=labels
    )
    df = df.sort_values(by=['f1-scores-test'], ascending=False)
    df.to_csv(path_or_buf=dump_file, sep='\t',float_format='%0.2f')


def classifaction_report_to_dict(report):
    report_data = {}
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split(' ')
        row_data = [r for r in row_data if r is not '']
        report_data[row_data[0]] = {
            'precision':float(row_data[1]),
            'recall':float(row_data[2]),
            'f1_score':float(row_data[3]),
            'support':float(row_data[4])
        }
    return report_data


def shuffleSplit_and_crossevaluate(
        data,
        target_names,
        tobescored_supplier:Callable[[], ToBeScoredAbstractClassifier],  #
        name = 'some_name',
        dump_path=None,
        n_splits=1,
        test_size=0.2,
        train_it = True,
        stratified = False

):
    # dt = TimeDiff()
    if stratified:
        splitter = StratifiedShuffleSplit(n_splits, test_size, random_state=111)
        # y_categorical = np.argmax(targets_binarized, axis=1)
        splits = splitter.split(X=range(len(data)), y=[d['target'][0] for d in data])
    else:
        splitter = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=111)
        splits = [(train,test) for train,test in splitter.split(X=range(len(data)))]

    # dt.print('start Crossevaluating')
    m_scores, std_scores = crosseval_mean_std_scores(
        tobescored_supplier,
        splits, data,
        scorer=partial(multilabel_scorer,
                       train_it = train_it,
                       target_names=target_names)
    )
    # pprint(m_scores)
    # print('PR-AUC-macro: train = %.2f; test = %.2f'%(m_scores['train']['PR-AUC-macro'],m_scores['test']['PR-AUC-macro']))
    # print('PR-AUC')
    # for label in target_names:
    #     print(label+': train = %0.3f; test = %0.3f'%(m_scores['train']['labelwise'][label]['PR-AUC'],m_scores['test']['labelwise'][label]['PR-AUC']))
    # dt.print('done evaluating')
    if dump_path is not None:
        if not os.path.exists(dump_path): os.makedirs(dump_path)
        scores_to_csv(m_scores,dump_path+'/'+name+'.csv')
    return m_scores