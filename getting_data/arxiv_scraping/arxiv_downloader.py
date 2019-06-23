import logging
import multiprocessing
import os
import sys
import traceback
from functools import partial
from time import sleep

import arxiv
from arxiv import slugify, urlretrieve


def download(obj,dirpath='./'):
    if dirpath[-1] != '/':
        dirpath += '/'
    file = slugify(obj) + '.pdf'
    path = dirpath + file
    out = None
    for k in range(10):
        try:
            urlretrieve(obj['pdf_url'], path)
            out = file
            break
        except Exception as e:
            sleep(k)
            # traceback.print_exc()
    if out is None:
        print(str(multiprocessing.current_process()) + 'could not download: %s' % file)
    return out

def has_pdf_url(obj):
    return obj.get('pdf_url', '')

def is_existent(d,files,k):
    file = slugify(d) + '.pdf'
    in_files = file in files
    if in_files:
        sys.stdout.write('\ralready got: %d files' % k)
        if k % 100 == 0:
            sys.stdout.flush()
    return in_files

if __name__ == '__main__':
    from pathlib import Path
    home = str(Path.home())
    # home = '/media/gdrive'
    # logger = arxiv.logger
    logging.basicConfig(filename='arxiv_downloader.log',
                                       filemode='a',
                                       format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                       datefmt='%H:%M:%S',
                                       level=logging.DEBUG)

    arxiv.logger = logging.getLogger('arxiv_downloader')
    # fh = logging.FileHandler('arxiv_downloader.log')
    # fh.setLevel(logging.DEBUG)
    # logger.addHandler(fh)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)

    result = arxiv.query(query="machine learning natural language processing", id_list=[], max_results=1000_000, sort_by="relevance",
                        sort_order="descending", prune=True, iterative=True, max_chunk_results=100)
    # arxiv.logger.handlers[0].flush()
    download_path=home+'/data/arxiv_papers/ml_nlp'
    if not os.path.isdir(download_path):
        os.mkdir(download_path)

    files = os.listdir(download_path)
    print('already got %d files'%len(files))

    # with multiprocessing.Pool(processes=16) as pool:
    g = (d for d in result() if has_pdf_url(d))
    g = (d for k,d in enumerate(g) if not is_existent(d,files,k))
    # g = (d for d in pool.imap_unordered(func=partial(download, dirpath=download_path), iterable=g) if d is not None)
    g = (download(d,download_path) for d in g if d is not None)
    for k,file in enumerate(g):
        sys.stdout.write('\r%d'%k)