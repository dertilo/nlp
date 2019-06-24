import os
from typing import Dict, List

import numpy
from commons import data_io


def getting_scierc_tagged_data(jsonl_file):

    def build_sentences(d:Dict):
        g = [tok for sent in d['sentences'] for tok in sent]
        spaced_tokens = [x for k,tok in enumerate(g) for x in [(tok,k),(' ',k+0.5)]]
        char_offsets = numpy.cumsum([0]+[len(x) for x,_ in spaced_tokens])
        tok2charoff = {k:char_offsets[i] for i,(tok,k) in enumerate(spaced_tokens)}
        text = ''.join([t for t,_ in spaced_tokens])

        spans = [(tok2charoff[token_start],tok2charoff[token_end+1],label) for sent_spans in d['ner'] for token_start,token_end,label in sent_spans]
        relations = [(tok2charoff[s1],tok2charoff[e1+1],tok2charoff[s2],tok2charoff[e2+1],label) for sent_rel in d['relations'] for s1,e1,s2,e2,label in sent_rel]
        return d['doc_key'],text,spans,relations

    return [build_sentences(d) for d in data_io.read_jsons_from_file(jsonl_file)]

def span_to_ann_line(idx,start,end,label,surface_text):
    s='T%d\t%s %d %d\t%s' % (idx, label, start, end,surface_text)
    return s

def to_rel_ann_line(idx,mentionId1,mentionId2,label):
    s='R%d\t%s Arg1:T%d Arg2:T%d' % (idx, label, mentionId1, mentionId2)
    return s

if __name__ == '__main__':
    data_path = '/home/tilo/code/NLP/scisci_nlp/data/scierc_data/json/'
    data = getting_scierc_tagged_data(data_path + 'dev.json')

    # for text,spans in data:
    #     for start,end,label in spans:
    #         print('%s - %s'%(text[start:end],label))
    path = './dingens'
    if not os.path.isdir(path):
        os.mkdir(path)

    for doc_name,text,spans,relations in data:
        data_io.write_to_file(path+'/'+doc_name+'.txt',[text])
        startend2Id = {'%d-%d'%(s,e):i for i,(s,e,l) in enumerate(spans)}
        def get_mention_id(s, e):
            return startend2Id['%d-%d' % (s, e)]
        lines = [span_to_ann_line(get_mention_id(s,e), s, e, l, text[s:e]) for s, e, l in spans]
        lines += [to_rel_ann_line(i,get_mention_id(s1,e1),get_mention_id(s2,e2),label) for i,(s1,e1,s2,e2,label) in enumerate(relations)]
        data_io.write_to_file(path+'/'+doc_name+'.ann',lines)