import os
import re

from commons import data_io

from getting_data.scholar import get_artices_from_googlescholar
from getting_data.scihub import SciHub

data_path = '/home/tilo/data/scihub_data/'


def dois_from_scholar_urls():
    sh = SciHub()
    no_doi_found_file = data_path + 'no_doi_found.txt'
    doi_not_in_scihub_file = data_path + 'doi_not_in_scihub.txt'
    no_doi_found = []
    doi_not_in_scihub = []
    shit_counter = 0
    pattern = re.compile('10.\d{4,9}/[-._;()/:A-Z0-9]+$')
    articles = get_artices_from_googlescholar(phrase='emotional geographies', max_num_hits=100)
    print('num hits %d' % len(articles))
    for art in articles:
        url = art.attrs['url'][0]
        if any([p in url for p in [
            'https://books.google.com/books'
        ]]):
            shit_counter += 1
            continue
        # url = 'https://journals.sagepub.com/doi/abs/10.1177/0891243287001002002'
        dois = [doi for doi in pattern.findall(url)]

        if len(dois) > 0:
            print(dois)
            for doi in dois:
                try:
                    sh.download(doi, destination=data_path)
                    print('got: %s' % doi)
                except:
                    doi_not_in_scihub.append(doi)
        else:
            no_doi_found.append(url)
    data_io.write_to_file(no_doi_found_file, no_doi_found, 'ab')
    data_io.write_to_file(doi_not_in_scihub_file, doi_not_in_scihub, 'ab')
    print('shit: %d' % shit_counter)

def scihub_by_title(
    title = 'The global and the intimate : feminism in our time'
    ):

    title_req = title.replace(' ','+')

    sh = SciHub()
    headers = {
                 'Content-Type':'application/x-www-form-urlencoded'
    }
    res = sh.sess.post('https://sci-hub.tw',data='sci-hub-plugin-check=&request=%s'%title_req,headers=headers)
    pattern = re.compile('10.\d{4,9}/[-._;()/:A-Z0-9]+')
    dois = set(pattern.findall(res.text))
    for doi in dois:
        sh.download(doi, destination=data_path)

if __name__ == '__main__':
    scihub_by_title()
    # dois_from_scholar_urls()

