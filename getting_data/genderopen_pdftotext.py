import os
import pdftotext

from commons import data_io

if __name__ == '__main__':
    from pathlib import Path
    home = str(Path.home())
    path_pdfs = home+"/data/gender_open_rep_pdfs"
    path_texts = home+"/data/gender_open_rep_texts"
    for i,file in enumerate(os.listdir(path_pdfs)):
        try:
            with open(path_pdfs + '/' + file, "rb") as f:
                pdf = pdftotext.PDF(f)
        except:
            continue
        pages = [page for page in pdf]
        data_io.write_to_file(path_texts+'/'+file.replace('.pdf','.txt'),pages)
        if i%100==0:
            print(i)
