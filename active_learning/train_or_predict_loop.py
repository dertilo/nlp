import json
import sys
import time

import spacy
import sqlalchemy
from sqlalchemy import Table, String, Column, select, create_engine, func
from sqlalchemy.ext.declarative import declarative_base

from active_learning.train_flair_seqtagger_from_postgres import build_sentences
from sequence_tagging.seq_tag_util import tags_to_token_spans
from sequence_tagging.spacy_features_sklearn_crfsuite import SpacyCrfSuiteTagger
from sqlalchemy_util.sqlalchemy_methods import bulk_update, fetchemany_sqlalchemy

annotator_machine = 'annotator_machine'

def row_to_dict(row):
    return {k:json.loads(v) if v is not None else None for k,v in row.items() }

def train_model():
    g = sqlalchemy_engine.execute(select([table]).where(sqlalchemy.not_(table.c.ner.like('%'+annotator_machine+'%'))))
    train_data = [sent for d in g for sent in build_sentences(row_to_dict(d),annotator_name='annotator_luan')]
    train_data = [[(token.text, token.tags['ner'].value) for token in datum] for datum in train_data]
    tagger = SpacyCrfSuiteTagger()
    tagger.fit(train_data)
    return tagger

def predict_on_db(model:SpacyCrfSuiteTagger):
    q = select([table])
    for batch in fetchemany_sqlalchemy(sqlalchemy_engine,q,batch_size=100):
        batch = [row_to_dict(d) for d in batch]
        sentences_flair = [(d['id'],sent_idx,sent) for d in batch for sent_idx,sent in enumerate(build_sentences(d))]
        data = [[token.text for token in sentence] for doc_id,sent_id,sentence in sentences_flair]
        pred_tags = model.predict(data)
        ner = {d['id']:[None for _ in range(len(d['sentences']))] for d in batch}
        for (doc_id, sent_id, _), tag_seq in zip(sentences_flair, pred_tags):
            ner[doc_id][sent_id] = tags_to_token_spans(tag_seq)

        def value_to_write(d,annotations):
            if isinstance(d['ner'],dict):
                d['ner'][annotator_machine]= annotations
            else:
                d['ner'] = {annotator_machine:annotations}

            return json.dumps(d['ner'])

        ids_values = [(json.dumps(d['id']),value_to_write(d,ner[d['id']])) for d in batch]
        with sqlalchemy_engine.connect() as conn:
            bulk_update(conn,table,'ner',ids_values)

        if traindata_significantly_changed():
            break

last_num_annotated_docs=[0]

def traindata_significantly_changed():
    q = select([func.count(table.c.id)]).where(sqlalchemy.or_(table.c.ner.like('%annotator_luan%'),table.c.ner.like('%annotator_human%')))
    num_annotated_docs = sqlalchemy_engine.execute(q).first()[0]
    significantly_changed = abs(num_annotated_docs-last_num_annotated_docs[0])/(1+last_num_annotated_docs[0])>0.1
    if significantly_changed:
        last_num_annotated_docs[0]=num_annotated_docs
    return significantly_changed

if __name__ == '__main__':
    sqlalchemy_base = declarative_base()
    ip = '10.1.1.29'
    sqlalchemy_engine = create_engine('postgresql://%s' % 'postgres:whocares@%s:5432/postgres'%ip)
    sqlalchemy_base.metadata.bind = sqlalchemy_engine

    table_name = 'scierc'
    columns = [Column('id', String, primary_key=True)] + [Column(colname, String) for colname in ['sentences','ner','relations','clusters','score']]
    table = Table(table_name, sqlalchemy_base.metadata, *columns, extend_existing=True)

    # score_col='score'
    # if score_col not in [c.name for c in table.columns]:
    #     add_column(sqlalchemy_engine,table,Column(score_col, String))
    #     print('added %s column'%score_col)

    spacy_nlp = spacy.load('en_core_web_sm', disable=['parser'])

    while True:
        while not traindata_significantly_changed():
            sys.stdout.write('\rno new train-data -> idling')
            time.sleep(1)
        # try:
        model = train_model()
        predict_on_db(model)
