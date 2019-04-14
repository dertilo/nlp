import sys
from time import sleep

import html2text
from Bio import Entrez

from commons import data_io


def pubmed_artice_generator(
        database='pubmed',
        query='heart',
        num_to_dump=9, batch_size=3):

    handle = Entrez.esearch(db=database, term=query, usehistory="Y", retmax=1000000)
    search_results = Entrez.read(handle)
    ids = search_results["IdList"]
    num_hits = len(ids)
    print("Found %d citations" % num_hits)
    num_to_dump = min(num_to_dump, num_hits)
    webenv = search_results["WebEnv"]
    query_key = search_results["QueryKey"]

    for start in range(0, num_to_dump, batch_size):
        print(start)
        fetch_handle = Entrez.efetch(db=database,
                                     # rettype="xml",
                                     # retmode="xml",
                                     retstart=start, retmax=batch_size,
                                     webenv=webenv, query_key=query_key)
        sleep(0.3)
        data = fetch_handle.read()
        # text = html2text.html2text(data)
        # print(text)
        fetch_handle.close()
        yield data


if __name__ == '__main__':

    data_io.write_to_file('./from_pubmed.xmls',pubmed_artice_generator())
