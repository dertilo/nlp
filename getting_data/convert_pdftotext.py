import os
import pdftotext
import sys

from commons import data_io

def convert_pdfs_to_text(path_pdfs,path_texts):
    if not os.path.isdir(path_texts):
        os.mkdir(path_texts)

    for i,file in enumerate(os.listdir(path_pdfs)):
        try:
            with open(path_pdfs + '/' + file, "rb") as f:
                pdf = pdftotext.PDF(f)
        except:
            continue
        pages = [page for page in pdf]
        data_io.write_to_file(path_texts+'/'+file.replace('.pdf','.txt'),pages)
        if i%100==0:
            sys.stdout.write('\r%d'%i)


if __name__ == '__main__':
    from pathlib import Path
    home = str(Path.home())
    # path_pdfs = home+"/data/gender_open_rep_pdfs"
    # path_texts = home+"/data/gender_open_rep_texts"
    convert_pdfs_to_text(
        '/media/tilo/77ceb20d-fc38-48bb-999e-8a03916f0e4e/data/arxiv_papers/ml_nlp',
        '/media/tilo/77ceb20d-fc38-48bb-999e-8a03916f0e4e/data/arxiv_papers/ml_nlp_texts',
                         )
