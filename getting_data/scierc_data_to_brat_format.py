import os
from typing import Dict, List

import numpy
from commons import data_io


def getting_scierc_tagged_data(jsonl_file):

    def build_sentences(d:Dict):
        offset=0
        g = [tok for sent in d['sentences'] for tok in sent]
        spaced_tokens = [x for k,tok in enumerate(g) for x in [(tok,k),(' ',k+0.5)]]
        char_offsets = numpy.cumsum([0]+[len(x) for x,_ in spaced_tokens])
        tokenidx2charoffsets = {k:char_offsets[i] for i,(tok,k) in enumerate(spaced_tokens)}
        text = ''.join([t for t,_ in spaced_tokens])

        spans = [(tokenidx2charoffsets[token_start],tokenidx2charoffsets[token_end+1],label) for sent_spans in d['ner'] for token_start,token_end,label in sent_spans]
        return d['doc_key'],text,spans

    return [build_sentences(d) for d in data_io.read_jsons_from_file(jsonl_file)]

def span_to_ann_line(idx,start,end,label,surface_text):
    s='T%d\t%s %d %d\t%s' % (idx, label, start, end,surface_text)
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

    for doc_name,text,spans in data:
        data_io.write_to_file(path+'/'+doc_name+'.txt',[text])
        lines = [span_to_ann_line(i,s,e,l,text[s:e]) for i,(s,e,l) in enumerate(spans)]
        data_io.write_to_file(path+'/'+doc_name+'.ann',lines)