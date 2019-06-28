import os
import subprocess

from bs4 import BeautifulSoup
from commons import data_io
from getting_data.parsing_pdfs.grobid_client_python.grobid_client import grobid_client


def process_file(file):
    with open(file) as f:
        xml = f.read()
        soup = BeautifulSoup(xml)
        raw_paragraphs = soup.find_all('p')
        refs = soup.find_all('ref')
        formula = soup.find_all('formula')
        list = soup.find_all('list')
        shit = [refs,formula,list]
        paragraphs = [' '.join([str(x) for x in paragraph if not any(x in sh for sh in shit)]) for paragraph in raw_paragraphs]
        data_io.write_to_file(processed_path+'/'+os.path.split(file)[1].replace('.xml','_processed.txt'),paragraphs)

if __name__ == '__main__':
    processed_path='/tmp/processed'
    xml_path = '/tmp/out'
    input_path = '/tmp/some_pdfs'
    if not os.path.isdir(processed_path): os.mkdir(processed_path)
    client = grobid_client(config_path='getting_data/parsing_pdfs/grobid_client_python/config.json')
    client.process(input_path, xml_path, 3, 'processFulltextDocument', False, False, False, False, False)

    # grobid_process = subprocess.Popen('')
    for file in os.listdir(xml_path):
        process_file(xml_path+'/'+file)
