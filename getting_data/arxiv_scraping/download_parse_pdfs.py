import sys
from io import StringIO

from commons import data_io

sys.path.append('.')
import json
import os

from arxiv import slugify
from sqlalchemy import Column, String, select

import getting_data.extract_fulltext_from_pdf.fulltext as fulltext_module
from sqlalchemy_util.sqlalchemy_base import sqlalchemy_base, sqlalchemy_engine
from sqlalchemy_util.sqlalchemy_methods import add_column, get_tables_by_reflection, insert_or_update

fulltext_module.log.propagate = False
fulltext_module.log.disabled = True

def parse_pdf_to_text(pdffile):
    content=None
    try:
    # old_stdout = sys.stdout
    # sys.stdout = StringIO()
        content = fulltext_module.fulltext(pdffile)
    # sys.stdout = old_stdout

    except Exception as e:
        print('could not parse %s' % pdffile)
    return content


if __name__ == '__main__':

    tables = get_tables_by_reflection(sqlalchemy_base.metadata,sqlalchemy_engine)
    arxiv_table = tables['arxiv']
    col_names = [c.name for c in arxiv_table._columns]
    pdffulltext = 'pdf_full_text'
    if pdffulltext not in col_names:
        add_column(sqlalchemy_engine,arxiv_table,Column(pdffulltext, String))
        print('added %s column'%pdffulltext)

    from pathlib import Path
    home = str(Path.home())
    download_path=home+'/data/arxiv_papers/ml_nlp'

    already_downloaded_files = os.listdir(download_path)

    with sqlalchemy_engine.connect() as conn:
        g = conn.execute(select([arxiv_table.c.id,arxiv_table.c.pdf_url,arxiv_table.c.title]).where(arxiv_table.c.pdf_full_text.is_(None)))
        for k,d in enumerate(g):
            # if k%100==0:
            print(k)
            d = {k:json.loads(v) for k,v in d.items()}
            file = slugify(d) + '.pdf'
            if file in already_downloaded_files:
                text = parse_pdf_to_text(download_path+'/'+file)
                if isinstance(text,str) and len(text)>0:
                    # print(file)
                    # data_io.write_to_file('/tmp/'+file+'.txt',[text])
                    q = arxiv_table.update().where(arxiv_table.c.id == json.dumps(d['id'])).values(**{pdffulltext: json.dumps(text)})
                    r = conn.execute(q)


'''
# count parsed pdf-files
SELECT count(*)
FROM 
(
SELECT DISTINCT t1.id
FROM public.arxiv t1
WHERE t1.pdf_full_text is not null
) dings;
'''