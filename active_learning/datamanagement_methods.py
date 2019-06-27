import json
import os
from typing import Iterable, Dict, List, Union, Any

import numpy
from commons import data_io
from sqlalchemy import create_engine, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Query

from sqlalchemy_util.sqlalchemy_base import get_sqlalchemy_base_engine
from sqlalchemy_util.sqlalchemy_methods import get_tables_by_reflection, bulk_update, fetchmany_sqlalchemy

DONE_ANNO = 'DoneAnnotating'
annotator_machine = 'annotator_machine'
annotator_luan = 'annotator_luan'
annotator_human = 'annotator_human'


def overwrite_ner_annotations(old_ner, annotations,annotator):
    if isinstance(old_ner,dict):
        old_ner[annotator]= annotations
    else:
        old_ner = {annotator:annotations}
    return old_ner

def row_to_dict(row):
    return {k:json.loads(v) if v is not None else None for k,v in row.items() }


def build_brat_lines(doc:Dict, annatators_of_interest=None):
    spaced_tokens, tok2charoff,tok2sent_id = spaced_tokens_and_tokenoffset2charoffset(doc['sentences'])
    text = ''.join([t for t,_ in spaced_tokens])

    def is_annotator_of_interest(annotator_name):
        if annatators_of_interest is None:
            return True
        elif isinstance(annatators_of_interest,list) and len(annatators_of_interest)>0:
            return annotator_name in annatators_of_interest
        else:
            assert False

    spans = build_ner_spans(doc, is_annotator_of_interest, tok2charoff)

    def get_mention_id(spans,annotator, end, start):
        x = [d['id'] for d in spans if
                             d['start'] == tok2charoff[start] and
                             d['end'] == tok2charoff[end + 0.5] and
                             d['annotator'] == annotator]
        assert len(x)==1
        return x[0]

    if doc['relations'] is not None and len(doc['relations'])>0:
        relations = [
            {
                'mention_id1': get_mention_id(spans,annotator, e1, s1),
                'mention_id2': get_mention_id(spans,annotator, e2, s2),
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

    file_name = doc['id']
    return file_name, text, spans, relations,attributes, notes


def build_ner_spans(doc, is_annotator_of_interest, tok2charoff):
    if doc['ner'] is None:
        doc['ner']={}
    spans = [
        {
            'start': tok2charoff[token_start],
            'end': tok2charoff[token_end + 0.5],
            'label': label,
            'ann_type': 'T',
            'annotator': annotator

        }
        for annotator, doc_spans in doc['ner'].items() if is_annotator_of_interest(annotator)
        for sent_spans in doc_spans
        for token_start, token_end, label in sent_spans]
    [d.update({'id': eid}) for eid, d in enumerate(spans)]
    return spans


def spaced_tokens_and_tokenoffset2charoffset(sentences:List[List[str]]):
    g = [(sent_id,tok) for sent_id,sent in enumerate(sentences) for tok in sent]
    spaced_tokens = [x for tok_id, (sent_id,tok) in enumerate(g) for x in [(tok, tok_id), (' ', tok_id + 0.5)]]
    tok2sent_id = {tok_id:sent_id for tok_id, (sent_id,tok) in enumerate(g)}
    char_offsets = numpy.cumsum([0] + [len(x) for x, _ in spaced_tokens])
    tok2charoff = {tok_id: char_offsets[i] for i, (tok, tok_id) in enumerate(spaced_tokens)}
    return spaced_tokens, tok2charoff,tok2sent_id

def charoffset2tokenoffset(sentences:List[List[str]]):
    _ ,tokenoffset2charoffset,_ = spaced_tokens_and_tokenoffset2charoffset(sentences)
    return {v:k for k,v in tokenoffset2charoffset.items() if round(k)==k}

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

def write_brat_annotations(q:Query,
                           path,
                           sqlalchemy_engine):
    if not os.path.isdir(path):
        os.mkdir(path)
    for d in sqlalchemy_engine.execute(q):
        doc = row_to_dict(d)
        write_brat_annotation(doc, path)


def write_brat_annotation(doc, path):
    doc_name, text, spans, relations, attributes, notes = build_brat_lines(doc)
    data_io.write_to_file(path + '/' + doc_name + '.txt', [text])
    lines = [span_to_ann_line(d, text) for d in spans]
    lines += [to_rel_ann_line(d) for d in relations]
    lines += [to_attr_ann_line(d) for d in attributes]
    lines += [to_notes_ann_line(d) for d in notes]
    ann_file = path + '/' + doc_name + '.ann'
    data_io.write_to_file(ann_file, lines)
    return ann_file


def parse_anno_line(line:str):
    out = {}
    if line.startswith('T'):
        ann_id,label_start_end,surface_text = line.split('\t')
        label, start, end  = label_start_end.split(' ')
        out = {'id':ann_id,'start':int(start),'end':int(end),'label':label,'surface_text':surface_text}

    elif line.startswith('A'):
        ann_id,type_refto_val = line.split('\t')
        attr_type, refto, val  = type_refto_val.split(' ')
        out = {'id':ann_id,'attr_type':attr_type,'refering_to':refto,'value':val}

    elif line.startswith('R'):
        ann_id,label_id1_id2 = line.split('\t')
        label, arg_id1, arg_id2  = label_id1_id2.split(' ')
        out = {'id':ann_id,'label':label,'id1':arg_id1.replace('Arg1:',''),'id2':arg_id2.replace('Arg2:','')}

    elif line.startswith('#'):
        ann_id,bla_refering_to,value = line.split('\t')
        _ , refering_to  = bla_refering_to.split(' ')
        out = {'id':ann_id,'refering_to':refering_to,'value':value}
    return out

import numpy as np

def build_ner(parsed_lines,char2tok:dict,tok2sent_id:dict):
    annotator2ids={}
    for d in parsed_lines:
        if d.get('attr_type', '') == 'Annotator':
            if d['value'] in annotator2ids.keys():
                annotator2ids[d['value']].append(d['refering_to'])
            else:
                annotator2ids[d['value']]=[d['refering_to']]

    def get_closest_tok(cid):
        char_offsets = [k for k in char2tok.keys()]
        argmin = np.argmin(np.abs(np.array(char_offsets) - cid))
        return char2tok[char_offsets[argmin]]

    def build_tokenoffsets(ids):
        sent_ids = set(tok2sent_id.values())
        annos = [[] for _ in range(len(sent_ids))]
        for d in parsed_lines:
            if d['id'].startswith('T') and d['id'] in ids and d['label']!= DONE_ANNO:
                start_tok = get_closest_tok(d['start'])
                end_tok = get_closest_tok(d['end'])-1
                annos[tok2sent_id[start_tok]].append([start_tok,end_tok,d['label']])

        return annos

    def ner_annotations_of_unkown_annotator(parsed_lines):
        sent_ids = set(tok2sent_id.values())
        annos = [[] for _ in range(len(sent_ids))]
        for d in parsed_lines:
            if d['id'].startswith('T') and d['label'] != DONE_ANNO:
                start_tok = get_closest_tok(d['start'])
                end_tok = get_closest_tok(d['end']) - 1
                annos[tok2sent_id[start_tok]].append([start_tok, end_tok, d['label']])
        return annos

    annos = ner_annotations_of_unkown_annotator(parsed_lines)

    d = {'annotator_%s' % a: build_tokenoffsets(ids) for a, ids in annotator2ids.items()}
    d[annotator_human]=annos
    return d

def join_all_ner_annotations(ner_anno:Dict[str, List], sentences:List[List[str]]):
    merged_spans=[[] for _ in range(len(sentences))]
    for annotator, sent_spans in ner_anno.items():
        assert len(sent_spans) == len(sentences)
        for sent_id, spans in enumerate(sent_spans):
            merged_spans[sent_id].extend(spans)
    return merged_spans

def parse_anno_lines(lines:List[str],sentences:List[List[str]]):
    _, _, tok2sent_id = spaced_tokens_and_tokenoffset2charoffset(sentences)
    char2tok_offsets = charoffset2tokenoffset(sentences)
    parsed_lines = [parse_anno_line(l) for l in lines]
    ner = build_ner(parsed_lines, char2tok_offsets, tok2sent_id)
    return {'ner':ner}


def unittest_parse_brat_annotations():
    # ip = '10.1.1.29'
    ip = 'localhost'
    sqlalchemy_base, sqlalchemy_engine = get_sqlalchemy_base_engine(ip=ip)
    table = get_tables_by_reflection(sqlalchemy_base.metadata, sqlalchemy_engine)['scierc']
    brat_path = './brat_configurations'
    # write_brat_annotations(select([table]).limit(3), brat_path, sqlalchemy_engine)
    for d in sqlalchemy_engine.execute(select([table]).limit(3)):
        doc = row_to_dict(d)
        ann_file = write_brat_annotation(doc, brat_path)
        _, _, tok2sent_id = spaced_tokens_and_tokenoffset2charoffset(doc['sentences'])

        anno = parse_anno_lines(data_io.read_lines(ann_file), doc['sentences'])

        assert (all([s1 == s2 and e1 == e2 and l1 == l2 and a1 == a2
                     for (a1, sents1), (a2, sents2) in zip(doc['ner'].items(), anno['ner'].items())
                     for x, y in zip(sents1, sents2)
                     for (s1, e1, l1), (s2, e2, l2) in zip(x, y)]))


def dump_table_to_jsonl(
        ip = '10.1.1.29',
        table_name='arxiv',
        dump_file='/tmp/arxiv.jsonl.gz',
        limit=100
    ):
    # ip = 'localhost'
    sqlalchemy_base, sqlalchemy_engine = get_sqlalchemy_base_engine(ip=ip)
    table = get_tables_by_reflection(sqlalchemy_base.metadata, sqlalchemy_engine)[table_name]
    q = select([table]).limit(limit)
    g = (row_to_dict(d) for batch in fetchmany_sqlalchemy(sqlalchemy_engine, q, batch_size=10000) for d in batch)
    data_io.write_jsons_to_file(dump_file, g)


if __name__ == '__main__':
    # unittest_parse_brat_annotations()
    dump_table_to_jsonl()
