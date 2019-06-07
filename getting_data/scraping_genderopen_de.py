import os
import sys
import time
import urllib
from urllib.request import urlopen as uReq  # Web client
from bs4 import BeautifulSoup
import re

from commons import data_io


def get_pdf_urls(query_keyword, page_number):
    page_url = "https://www.genderopen.de/discover?rpp=100&etal=0&query=%s&scope=/&group_by=none&page=%d&sort_by=score&order=desc" % (
    query_keyword, page_number)
    uClient = uReq(page_url)
    page_soup = BeautifulSoup(uClient.read(), "html.parser")
    uClient.close()
    pattern = re.compile('http://www.genderopen\.de.{1,400}\.pdf')
    page_str = str(page_soup)
    filenames = pattern.findall(page_str)
    return filenames

if __name__ == '__main__':
    data_dir = '/home/tilo/data/gender_open_rep_pdfs'
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    urls_file = data_dir + '/urls.txt'
    if os.path.isfile(urls_file):
        already_downloaded_urls = list(data_io.read_lines(urls_file))
    else:
        already_downloaded_urls = []

    for query_keyword in ['Geschlecht','Eine','theorie','theory']:#'women','Frau'
        for page_number in range(1,200000):
            while True:
                try:
                    filenames = get_pdf_urls(query_keyword, page_number)
                    break
                except Exception:
                    print('retrying query: %s; page: %d'%(query_keyword,page_number))
                    time.sleep(5)

            if len(filenames)==0:
                break

            new_urls = [p for p in filenames if p not in already_downloaded_urls]
            print('found %d new urls' % len(new_urls))
            already_downloaded_urls.extend(new_urls)
            for i,p in enumerate(new_urls):
                os.system('wget -N -b -q -P %s %s'%(data_dir,p))
                time.sleep(0.2)
                sys.stdout.write('\r'); sys.stdout.flush()
                sys.stdout.write('\rnum-current page: %d'%i); sys.stdout.flush()
            data_io.write_to_file(urls_file,new_urls,'ab')

            if len(already_downloaded_urls)%100==0:
                print('got %d files' % len(already_downloaded_urls))