import os
import sys

import arxiv
from arxiv import slugify,urlretrieve

def download_if_not_existent(obj,files, dirpath='./', slugify=slugify):
    if not obj.get('pdf_url', ''):
        print("Object has no PDF URL.")
        return
    if dirpath[-1] != '/':
        dirpath += '/'
    file = slugify(obj) + '.pdf'
    if file not in files:
        path = dirpath + file
        urlretrieve(obj['pdf_url'], path)
        return path

if __name__ == '__main__':
    # TODO: started 12:45 ... got ~100 after 15 minutes
    '''
    12:45 - start
    13:00 - 100
    14:00 - 480
    
    '''
    from pathlib import Path
    home = str(Path.home())

    result = arxiv.query(query="machine learning natural language processing", id_list=[], max_results=None, sort_by="relevance",
                        sort_order="descending", prune=True, iterative=True, max_chunk_results=100)
    download_path=home+'/data/arxiv_papers/ml_nlp'
    if not os.path.isdir(download_path):
        os.mkdir(download_path)

    files = os.listdir(download_path)
    for k,paper in enumerate(result()):
        sys.stdout.write('\ralready got: %d'%k)
        download_if_not_existent(paper,files,download_path)