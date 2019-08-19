import hashlib
import os
import subprocess
from pprint import pprint
from time import time
from typing import Iterable, Generator, List, Dict

import numpy as np
from scipy.sparse import csr_matrix, vstack
import itertools

class TimeDiff(object):
    def __init__(self,prefix="start"):
        self.now = time()
        self.prefix =prefix
    def get(self):
        dT = time() -self.now
        self.now = time()
        return dT
    def print(self,prefix):
        print(self.prefix+' => '+str('{0:.3g}'.format(self.get()))+" secs => "+prefix)
        self.prefix = prefix

def get_dict_paths(paths, root_path, my_dict):
    if not isinstance(my_dict, dict):
        paths.append(root_path)
        return root_path
    for k, v in my_dict.items():
        path = root_path + [k]
        get_dict_paths(paths, path, v)

def get_val(d,path):
    for p in path:
        d = d.get(p)
    return d

def set_val(d,path,value):
    for i in range(len(path)-1):
        p = path[i]
        if p in d:
            d = d.get(p)
        else:
            d.__setitem__(p,{})
            d = d.get(p)
    d.__setitem__(path[-1],value)

def run_jar(command,java_home = '/usr/lib/jvm/java-8-oracle'):
    environ = os.environ.copy()
    environ['JAVA_HOME'] = java_home
    p = subprocess.Popen(command, env=environ, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    returned = []
    while p.poll() is None:
        returned.append(p.stdout.readline())  # This blocks until it receives a newline.
        returned.append(p.stderr.readline()) # This blocks until it receives a newline.
    return returned


def exec_command(command):
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {'stdout': p.stdout.readlines(),'stderr':p.stderr.readlines()}


def cos_matrix_multiplication(matrix, vector):
    """
    Calculating pairwise cosine distance using matrix vector multiplication.
    """
    dotted = matrix.dot(vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)
    matrix_vector_norms = np.multiply(matrix_norms, vector_norm)
    neighbors = np.divide(dotted, matrix_vector_norms)
    return neighbors

def todict(obj, classkey=None):
    if isinstance(obj, dict):
        data = {}
        for (k, v) in obj.items():
            data[k] = todict(v, classkey)
        return data
    elif hasattr(obj, "_ast"):
        return todict(obj._ast())
    elif isinstance(obj,list): #hasattr(obj, "__iter__") and not isinstance(obj,str):
        return [todict(v, classkey) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dict([(key, todict(value, classkey))
            for key, value in obj.__dict__.items()
            if not callable(value) and not key.startswith('_')])
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    else:
        return obj

def dicts_to_csr(dicts:List[Dict], num_rows=None, num_cols=None):
    assert isinstance(dicts,list) and len(dicts)>0
    assert all([isinstance(d,dict) for d in dicts])
    g = [(row, int(col),val) for row,d in enumerate(dicts) for col,val in d.items()]
    col = [t[1] for t in g]
    values = [t[2] for t in g]
    row = [t[0] for t in g]
    num_dim = max(col)+1
    shape = (num_rows if num_rows is not None else len(dicts),
             num_cols if num_cols is not None else num_dim)
    return csr_matrix((values, (row, col)), shape=shape)

def csr_vectors_to_dicts(vects:List[csr_matrix]):
    csr = vstack(vects, format='csr')
    return csr_to_dicts(csr)

def ndarray_to_dicts(x:np.ndarray,dim_names=None,filter_on_val=lambda v:True):
    if dim_names is None:
        dim_names = [i for i in range(x.shape[1])]
    return [{dim_names[col]:val for col,val in enumerate(row) if filter_on_val(val)} for row in x]

def csr_to_dicts(x:csr_matrix,dim_names=None):
    if dim_names is None:
        dim_names = [i for i in range(x.shape[1])]
    vert_idx,horiz_idx = x.nonzero()
    return [{dim_names[k]:v for k,v in zip(horiz_idx[np.where(vert_idx==row_idx)],x.data[np.where(vert_idx==row_idx)])} for row_idx in range(x.shape[0])]

def process_batchwise(process_fun, iterable:Iterable, batch_size=1024):
    return (d for batch in iterable_to_batches(iterable, batch_size) for d in process_fun(batch))

def consume_batchwise(consume_fun, iterable:Iterable, batch_size=1024):
    for batch in iterable_to_batches(iterable, batch_size):
        consume_fun(batch)

def iterable_to_batches(g:Iterable, batch_size)->Generator[List[object], None, None]:
    g = iter(g) if isinstance(g,list) else g
    batch = []
    while True:
        try:
            batch.append(next(g))
            if len(batch)==batch_size:
                yield batch
                batch = []
        except StopIteration as e: # there is no next element in iterator
            break
    if len(batch)>0:
        yield batch

def hash_list_of_strings(l:List[str]):
    return hashlib.sha1('_'.join(l).encode('utf-8')).hexdigest()

if __name__ == '__main__':
    dim_names = ['a','b','c']
    x = dicts_to_csr([{'a':0.1,'c':0.2},{'c':0.2}],dim_names)
    print(x)