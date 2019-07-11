from time import time
from typing import Dict
from hyperopt import Trials, STATUS_OK, fmin, tpe


def tune_hyperparams(
        data_supplier,
        search_space:Dict,
        score_fun,
        max_evals = 9,
        trials = Trials(),
        metric_for_hyperopt='accuracy',
        n_jobs=5
):
    data = data_supplier()

    def objective(params_dict):
        start = time()
        metrics = score_fun(data,**params_dict)
        score = metrics[metric_for_hyperopt]
        return {'loss': -score,
                'status': STATUS_OK,
                'mean-metrics':metrics,
                'crossval_duration_in_s': time()-start,
                'hyperparams': params_dict
                }

    # start = time()
    _ = fmin(objective, search_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    # duration = time() -start
    min_loss,best_hyperparams = min([(t['loss'],t['hyperparams']) for t in trials.results],key=lambda x:x[0])
    return best_hyperparams,trials