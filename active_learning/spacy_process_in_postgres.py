import json
from multiprocessing.pool import Pool
from time import time

import spacy
from sqlalchemy import select, Column, String

from pytorch_util.pytorch_methods import iterate_and_time
from sqlalchemy_util.sqlalchemy_base import get_sqlalchemy_base_engine
from sqlalchemy_util.sqlalchemy_methods import get_tables_by_reflection, insert_or_update, fetchmany_sqlalchemy, \
    process_table_batchwise, add_column

def initializer_fun(spacy_model_name='en_core_web_sm'):
    global spacy_nlp
    spacy_nlp = spacy.load(spacy_model_name)

def process_fun(batch):
    g = (json.loads(d['pdf_full_text']) for d in batch)
    spacy_docs = [(d['id'], spacy_doc) for d,spacy_doc in zip(batch,spacy_nlp.pipe(g,batch_size=len(batch)))]
    processed_docs = [{'id': eid.replace('/', '').replace(':', ''), tok_sent_col: process_to_str(doc)} for
                      eid, doc in
                      spacy_docs]
    return processed_docs

if __name__ == '__main__':
    ip = '10.1.1.29'
    # ip = 'localhost'
    sqlalchemy_base,sqlalchemy_engine = get_sqlalchemy_base_engine(ip=ip)

    arxiv_table = get_tables_by_reflection(sqlalchemy_base.metadata,sqlalchemy_engine)['arxiv']
    # scierc_table = get_tables_by_reflection(sqlalchemy_base.metadata,sqlalchemy_engine)['scierc']
    tok_sent_col = 'tokenized_sentences'
    # if tok_sent_col not in [c.name for c in arxiv_table.columns]:
    #     add_column(sqlalchemy_engine,arxiv_table,Column(tok_sent_col, String))
    #     print('added %s column'%tok_sent_col)


    q = select([arxiv_table.c.id,arxiv_table.c.pdf_full_text]).where(arxiv_table.c.pdf_full_text.isnot(None)).limit(30)
    def process_to_str(doc):
        return json.dumps([[tok.text for tok in sent]for sent in doc.sents])
    batch_size = 14
    pure_spacy_time=0
    start = time()
    with sqlalchemy_engine.connect() as conn:
        g = (batch for batch in fetchmany_sqlalchemy(sqlalchemy_engine, q, batch_size=batch_size))
        with Pool(processes=7, initializer=initializer_fun, initargs=('en_core_web_sm',)) as pool:
            for processed_batch,dur in iterate_and_time(pool.imap_unordered(process_fun, g)):
                pure_spacy_time+= dur
                insert_or_update(conn, arxiv_table, [tok_sent_col], processed_batch)
    overall_dur = time() - start
    print('old method multithreaded took: %0.2f secs;spacy took: %0.2f secs; postgres: %0.2f' % (overall_dur, pure_spacy_time,overall_dur-pure_spacy_time))

    start = time()
    initializer_fun('en_core_web_sm')
    pure_spacy_time = process_table_batchwise(sqlalchemy_engine, q, arxiv_table, process_fun, batch_size=batch_size, num_processes=0)
    overall_dur = time() - start
    print('new method took: %0.2f secs;spacy took: %0.2f secs; postgres: %0.2f' % (overall_dur, pure_spacy_time,overall_dur-pure_spacy_time))

    start = time()
    pure_spacy_time = process_table_batchwise(sqlalchemy_engine, q, arxiv_table, process_fun, batch_size=batch_size, num_processes=7, initializer_fun=initializer_fun, initargs=('en_core_web_sm',))
    overall_dur = time() - start
    print('new method multithreaded took: %0.2f secs;spacy took: %0.2f secs; postgres: %0.2f' % (overall_dur, pure_spacy_time,overall_dur-pure_spacy_time))

