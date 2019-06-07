import os
import re

from commons import data_io

from getting_data.scholar import get_artices_from_googlescholar
from getting_data.scihub import SciHub

if __name__ == '__main__':

    sh = SciHub()
    data_path = '/home/tilo/data/scihub_data/'
    no_doi_found_file = data_path + 'no_doi_found.txt'
    doi_not_in_scihub_file = data_path + 'doi_not_in_scihub.txt'

    no_doi_found = []

    doi_not_in_scihub = []

    shit_counter=0
    pattern = re.compile('10.\d{4,9}/[-._;()/:A-Z0-9]+$')
    articles = get_artices_from_googlescholar(phrase='emotional geographies', max_num_hits=100)
    print('num hits %d'%len(articles))
    for art in articles:
        url = art.attrs['url'][0]
        if any([p in url for p in [
            'https://books.google.com/books'
        ]]):
            shit_counter+=1
            continue
    # url = 'https://journals.sagepub.com/doi/abs/10.1177/0891243287001002002'
        dois = [doi for doi in pattern.findall(url)]

        if len(dois)>0:
            print(dois)
            for doi in dois:
                try:
                    sh.download(doi, destination=data_path)
                    print('got: %s'%doi)
                except:
                    doi_not_in_scihub.append(doi)
        else:
            no_doi_found.append(url)

    data_io.write_to_file(no_doi_found_file, no_doi_found, 'ab')
    data_io.write_to_file(doi_not_in_scihub_file, doi_not_in_scihub, 'ab')

    print('shit: %d'%shit_counter)
    # retrieve 5 articles on Google Scholars related to 'bittorrent'
    # results = sh.search('feminism', 5)
    # print('found %d papers'%len(results))
    #
    # for paper in results['papers']:
    #     print(paper['url'])
    #     sh.download(paper['url'],destination='/home/tilo/data/scihub_data/')