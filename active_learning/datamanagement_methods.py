import json
import os
from typing import Iterable, Dict

import numpy
from commons import data_io
from sqlalchemy import create_engine, select
from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy_util.sqlalchemy_methods import get_tables_by_reflection


def row_to_dict(row):
    return {k:json.loads(v) if v is not None else None for k,v in row.items() }


def build_brat_lines(doc:Dict, annatators_of_interest=None):
    g = [tok for sent in doc['sentences'] for tok in sent]
    spaced_tokens = [x for k,tok in enumerate(g) for x in [(tok,k),(' ',k+0.5)]]
    char_offsets = numpy.cumsum([0]+[len(x) for x,_ in spaced_tokens])
    tok2charoff = {k:char_offsets[i] for i,(tok,k) in enumerate(spaced_tokens)}
    text = ''.join([t for t,_ in spaced_tokens])

    def is_annotator_of_interest(annotator_name):
        if annatators_of_interest is None:
            return True
        elif isinstance(annatators_of_interest,list) and len(annatators_of_interest)>0:
            return annotator_name in annatators_of_interest
        else:
            assert False

    spans = [
        {
            'start':tok2charoff[token_start],
            'end': tok2charoff[token_end+0.5],
            'label':label,
            'ann_type':'T',
            'annotator':annotator

        }
        for annotator,doc_spans in doc['ner'].items() if is_annotator_of_interest(annotator)
        for sent_spans in doc_spans
        for token_start,token_end,label in sent_spans]
    [d.update({'id':eid}) for eid,d in enumerate(spans)]

    def get_mention_id(annotator, end, start):
        x = [d['id'] for d in spans if
                             d['start'] == tok2charoff[start] and
                             d['end'] == tok2charoff[end + 0.5] and
                             d['annotator'] == annotator]
        assert len(x)==1
        return x[0]
    if doc['relations'] is not None and len(doc['relations'])>0:
        relations = [
            {
                'mention_id1': get_mention_id(annotator, e1, s1),
                'mention_id2': get_mention_id(annotator, e2, s2),
                'label': label,
                'ann_type': 'R',
                'annotator':annotator
            }

            for annotator,doc_rel in doc['relations'].items() if is_annotator_of_interest(annotator)
            for sent_rel in doc_rel
            for s1,e1,s2,e2,label in sent_rel]
        [d.update({'id':eid}) for eid,d in enumerate(relations)]
    else:
        relations = []
    # attributes not working for relations!!
    attributes = [{'ann_type':'A','id':eid,'refering_to':'%s%d'%(d['ann_type'],d['id']),'attribute_type':'Annotator','value':d['annotator'].replace('annotator_','')} for eid,d in enumerate(spans)]
    notes = [{'id':eid,'refering_to':'%s%d'%(d['ann_type'],d['id']),'value':'annotator=%s'%d['annotator'].replace('annotator_','')} for eid,d in enumerate(spans+relations)]

    file_name = doc['id'].replace('/','').replace(':','')
    return file_name, text, spans, relations,attributes, notes



def span_to_ann_line(d,text):
    s='T%d\t%s %d %d\t%s' % (d['id'], d['label'], d['start'], d['end'],text[d['start']:d['end']])
    return s

def to_rel_ann_line(d):
    s='R%d\t%s Arg1:T%d Arg2:T%d' % (d['id'], d['label'], d['mention_id1'], d['mention_id2'])
    return s

def to_attr_ann_line(d):
    s='A%d\t%s %s %s' % (d['id'], d['attribute_type'],d['refering_to'],d['value'])
    return s
def to_notes_ann_line(d):
    s='#%d\tAnnotatorNotes %s\t%s' % (d['id'], d['refering_to'],d['value'])
    return s

if __name__ == '__main__':
    sqlalchemy_base = declarative_base()
    ip = '10.1.1.29'
    sqlalchemy_engine = create_engine('postgresql://%s' % 'postgres:whocares@%s:5432/postgres'%ip)
    sqlalchemy_base.metadata.bind = sqlalchemy_engine
    table = get_tables_by_reflection(sqlalchemy_base.metadata,sqlalchemy_engine)['scierc']

    g = sqlalchemy_engine.execute(select([table]).limit(3))

    path = './brat_configurations'
    if not os.path.isdir(path):
        os.mkdir(path)

    for d in g:
        doc_name, text, spans, relations, attributes,notes = build_brat_lines(row_to_dict(d))
        data_io.write_to_file(path+'/'+doc_name+'.txt',[text])

        lines = [span_to_ann_line(d, text) for d in spans]
        lines += [to_rel_ann_line(d) for d in relations]
        lines += [to_attr_ann_line(d) for d in attributes]
        lines += [to_notes_ann_line(d) for d in notes]
        data_io.write_to_file(path+'/'+doc_name+'.ann',lines)



