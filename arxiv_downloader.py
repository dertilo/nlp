import os

import arxiv

if __name__ == '__main__':
    from pathlib import Path
    home = str(Path.home())

    result = arxiv.query(query="machine learning natural language processing", id_list=[], max_results=None, sort_by="relevance",
                        sort_order="descending", prune=True, iterative=True, max_chunk_results=100)
    download_path=home+'/data/arxiv_papers/ml_nlp'
    if not os.path.isdir(download_path):
        os.mkdir(download_path)

    for paper in result():
        arxiv.download(paper,download_path)