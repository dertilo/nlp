import json
import sys
sys.path.append('.')
import time

import spacy
import sqlalchemy
from sqlalchemy import Table, String, Column, select, func

from active_learning.datamanagement_methods import row_to_dict, annotator_luan, annotator_human, \
    overwrite_ner_annotations, annotator_machine
from active_learning.train_flair_seqtagger_from_postgres import build_sentences
from sequence_tagging.seq_tag_util import tags_to_token_spans
from sequence_tagging.spacy_features_sklearn_crfsuite import SpacyCrfSuiteTagger
from sqlalchemy_util.sqlalchemy_base import get_sqlalchemy_base_engine
from sqlalchemy_util.sqlalchemy_methods import process_table_batchwise


def train_model():
    g = sqlalchemy_engine.execute(select([table])
                                  .where(sqlalchemy.or_(table.c.ner.like('%' + annotator_human + '%'),
                                                        table.c.ner.like('%' + annotator_luan + '%'))))
    train_data = [sent for d in g for sent in build_sentences(row_to_dict(d), annotator_names=[annotator_human,annotator_luan])]
    train_data = [[(token.text, token.tags['ner'].value) for token in datum] for datum in train_data]
    print('training on %d samples'%len(train_data))
    if len(train_data)>0:
        tagger = SpacyCrfSuiteTagger()
        tagger.fit(train_data)
    else:
        tagger = None
    return tagger

def predict_on_db(model:SpacyCrfSuiteTagger):
    q = select([table]) # here maybe some ranking
    def process_fun(batch):
        batch = [row_to_dict(d) for d in batch]
        sentences_flair = [(d['id'],sent_idx,sent) for d in batch for sent_idx,sent in enumerate(build_sentences(d))]
        data = [[token.text for token in sentence] for doc_id,sent_id,sentence in sentences_flair]
        pred_tags = model.predict(data)
        docid2sent_spans = group_by_doc_and_sent_ids_convert_tags2spans(batch, pred_tags, sentences_flair)

        processed_batch = [{'id':json.dumps(d['id']),'ner':json.dumps(overwrite_ner_annotations(d['ner'], docid2sent_spans[d['id']],annotator_machine))} for d in batch]
        return processed_batch

    process_table_batchwise(sqlalchemy_engine, q, table, process_fun, batch_size=100,stop_fun=traindata_significantly_changed)


def group_by_doc_and_sent_ids_convert_tags2spans(batch, pred_tags, sentences_flair):
    docid2sent_spans = {d['id']: [None for _ in range(len(d['sentences']))] for d in batch}
    for (doc_id, sent_id, _), tag_seq in zip(sentences_flair, pred_tags):
        docid2sent_spans[doc_id][sent_id] = tags_to_token_spans(tag_seq)
    return docid2sent_spans


last_num_annotated_docs=[0]

def traindata_significantly_changed():
    q = select([func.count(table.c.id)]).where(sqlalchemy.or_(table.c.ner.like('%annotator_luan%'),table.c.ner.like('%annotator_human%')))
    num_annotated_docs = sqlalchemy_engine.execute(q).first()[0]
    significantly_changed = abs(num_annotated_docs-last_num_annotated_docs[0])/(1+last_num_annotated_docs[0])>0.1
    if significantly_changed:
        last_num_annotated_docs[0]=num_annotated_docs
    return significantly_changed

if __name__ == '__main__':
    start = time.time()
    ip = '10.1.1.29'
    # ip = 'localhost'
    sqlalchemy_base,sqlalchemy_engine = get_sqlalchemy_base_engine(ip=ip)

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

        model = train_model()
        if model is not None:
            predict_on_db(model)
