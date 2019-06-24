import json

import spacy
import sqlalchemy
from flair.models import SequenceTagger
from sqlalchemy import Table, String, Column, select, create_engine
from sqlalchemy.ext.declarative import declarative_base
from torch.utils.data import Dataset

from sqlalchemy_util.sqlalchemy_methods import get_tables_by_reflection, fetchemany_sqlalchemy, insert_or_update

if __name__ == '__main__':
    sqlalchemy_base = declarative_base()
    ip = '10.1.1.29'
    sqlalchemy_engine = create_engine('postgresql://%s' % 'postgres:whocares@%s:5432/postgres'%ip)
    sqlalchemy_base.metadata.bind = sqlalchemy_engine

    spacy_nlp = spacy.load('en_core_web_sm')

    arxiv_table = get_tables_by_reflection(sqlalchemy_base.metadata,sqlalchemy_engine)['arxiv']
    scierc_table = get_tables_by_reflection(sqlalchemy_base.metadata,sqlalchemy_engine)['scierc']

    q = select([arxiv_table.c.id,arxiv_table.c.pdf_full_text]).where(arxiv_table.c.pdf_full_text.isnot(None))
    def process_to_str(doc):
        return json.dumps([[tok.text for tok in sent]for sent in doc.sents])

    with sqlalchemy_engine.connect() as conn:
        for batch in fetchemany_sqlalchemy(sqlalchemy_engine, q, batch_size=10):
            spacy_docs = [(d['id'], spacy_nlp(json.loads(d['pdf_full_text']))) for d in batch]
            ids_sentences = [{'id':eid,'sentences':process_to_str(doc)} for eid, doc in spacy_docs]
            insert_or_update(conn,scierc_table,['sentences'],ids_sentences)
            break


