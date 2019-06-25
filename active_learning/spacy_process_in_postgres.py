import json
from time import time

import spacy
from sqlalchemy import select
from sqlalchemy_util.sqlalchemy_base import get_sqlalchemy_base_engine
from sqlalchemy_util.sqlalchemy_methods import get_tables_by_reflection, insert_or_update, fetchmany_sqlalchemy, \
    update_batchwise

if __name__ == '__main__':
    # ip = '10.1.1.29'
    ip = 'localhost'
    sqlalchemy_base,sqlalchemy_engine = get_sqlalchemy_base_engine(ip=ip)

    spacy_nlp = spacy.load('en_core_web_sm')

    arxiv_table = get_tables_by_reflection(sqlalchemy_base.metadata,sqlalchemy_engine)['arxiv']
    # scierc_table = get_tables_by_reflection(sqlalchemy_base.metadata,sqlalchemy_engine)['scierc']

    q = select([arxiv_table.c.id,arxiv_table.c.pdf_full_text]).where(arxiv_table.c.pdf_full_text.isnot(None)).limit(100)
    def process_to_str(doc):
        return json.dumps([[tok.text for tok in sent]for sent in doc.sents])
    batch_size = 100

    def process_fun(batch):
        spacy_docs = [(d['id'], spacy_nlp(json.loads(d['pdf_full_text']))) for d in batch]
        processed_docs = [{'id': eid.replace('/', '').replace(':', ''), 'tokenized_sentences': process_to_str(doc)} for
                          eid, doc in
                          spacy_docs]
        return processed_docs

    start = time()
    with sqlalchemy_engine.connect() as conn:
        for batch in fetchmany_sqlalchemy(sqlalchemy_engine, q, batch_size=10):
            insert_or_update(conn,arxiv_table,['tokenized_sentences'],process_fun(batch))
    print('old method took: %0.2f secs'%(time()-start))

    start = time()
    update_batchwise(sqlalchemy_engine, q, arxiv_table, process_fun, batch_size=batch_size)
    print('new method took: %0.2f secs'%(time()-start))

