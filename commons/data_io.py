import gzip
import json
from time import time

import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, List, Iterable
import os

def read_data_build_sparse_matrix(dataFilePath):
    counter=0; triples=[]; targets=[]
    with gzip.open(dataFilePath,'rt') as f:
        for line in f:
            s = line.split('\t')

            target = s[0].split(',')
            index_value_map = json.loads(s[1])
            targets.append(target)
            # data.append(datum)
            for featureIndex,value in index_value_map.items():
                triples.append((value,counter,float(featureIndex)))
            counter+=1


    values = [t[0] for t in triples]
    row = [t[1] for t in triples]
    col = [t[2] for t in triples]

    num_dim = max(col)+1
    start_time = time()
    X = csr_matrix((values, (row, col)), shape=(counter, num_dim))
    print('building sparse matrix of shape: '+str(X.shape) +' took: ' + str(time() - start_time))
    return X,targets


def write_jsons_to_file(file:str,data:Iterable, mode="wb"):
    def process_line(hit):
        try:
            line = json.dumps(hit,skipkeys=True, ensure_ascii=False)
            line = line + "\n"
            line = line.encode('utf-8')
        except Exception as e:
            line=''
            print(e)
        return line
    with gzip.open(file, mode=mode) if file.endswith('gz') else open(file, mode=mode) as f:
        f.writelines((process_line(d) for d in data))

def write_json_to_file(file:str,datum:Dict,mode="wb"):
    with gzip.open(file, mode=mode) if file.endswith('gz') else open(file, mode=mode) as f:
        line = json.dumps(datum,skipkeys=True, ensure_ascii=False)
        line = line.encode('utf-8')
        f.write(line)


def write_jsons_to_files(files:Iterable[str], data:Iterable[Dict], mode="wb"):
    [write_json_to_file(file,datum,mode) for file,datum in zip(files,data)]

def write_to_file(file,lines=None, mode='wb'):
    def process_line(line):
        line= line + '\n'
        return line.encode('utf-8')
    with gzip.open(file, mode=mode) if file.endswith('.gz') else open(file,mode=mode) as f:
        f.writelines((process_line(l) for l in lines))

def read_lines_from_files(path, mode ='b', encoding ='utf-8', limit=np.Inf):
    g = (line for file in os.listdir(path) for line in read_lines(os.path.join(path,file),mode,encoding))
    c=0
    for line in g:
        c+=1
        if c>limit:break
        yield line

def read_lines(file, mode ='b', encoding ='utf-8', limit=np.Inf):
    counter = 0
    with gzip.open(file, mode='r'+mode) if file.endswith('.gz') else open(file,mode='r'+mode) as f:
        for line in f:
            counter+=1
            if counter>limit:
                break
            if mode == 'b':
                yield line.decode(encoding).replace('\n','')
            elif mode == 't':
                yield line.replace('\n','')

def read_jsons_from_file_to_list(file, limit=np.Inf):
    data = []; counter = 0
    for d in read_jsons_from_file(file):
        data.append(d)
        counter+=1
        if counter > limit:
            break
    return data

def read_jsons_from_file(file, limit=np.Inf):
    with gzip.open(file, mode="rb") if file.endswith('.gz') else open(file, mode="rb") as f:
        counter=0
        for line in f:
            # assert isinstance(line,bytes)
            counter += 1
            if counter > limit: break
            yield json.loads(line.decode('utf-8'))