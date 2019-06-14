import os

from commons import data_io
from commons.util_methods import iterable_to_batches

from getting_data.clef2019 import get_Clef2019_data

if __name__ == '__main__':
    import spacy
    nlp = spacy.load('en')

    # Add neural coref to SpaCy's pipe
    import neuralcoref
    neuralcoref.add_to_pipe(nlp)

    path = '/home/tilo/code/misc/clef2019-factchecking-task1/data/training'
    dump_path = '/home/tilo/data/coref'
    debate_lines = get_Clef2019_data(path)
    for k,batch in enumerate(iterable_to_batches([d['utterance'] for d in debate_lines],batch_size=32)):
        k+=1
        text = ' '.join(batch)
        doc = nlp(text)
        data_io.write_to_file(dump_path+'/text_%d.txt'%k,[text])
        m_id = [0]
        def get_inc_m_id():
            val = m_id[0]
            m_id[0]+=1
            return val


        standoff_corefs = []
        for cluster_id,cluster in enumerate(doc._.coref_clusters):
            mentions = [(s.start_char,s.end_char,s.text) for s in cluster]
            cluster_mention = (get_inc_m_id(),cluster.main.start_char,cluster.main.end_char,cluster.main.text)
            mentions = [(get_inc_m_id(),start,end,text) for start,end,text in mentions if text != cluster.main.text and sum([s<=start and e>=end for s,e,t in mentions])==1]
            if len(mentions) > 0:
                standoff_coref = [
                    'T%d\tClusterCenter %d %d\t%s'%(cluster_mention[0],cluster_mention[1],cluster_mention[2],cluster_mention[3]),
                    'E%d\tClusterCenter:T%d'%(cluster_id,cluster_mention[0]),
                    '*\tCoref E%d '%cluster_id+' '.join(['T%d'%m[0] for m in mentions])
                                  ]+['T%d\tMention %d %d\t%s'%(mid,s,e,t) for mid,s,e,t in mentions]
                standoff_corefs.extend(standoff_coref)
        data_io.write_to_file(dump_path+'/text_%d.ann' % k, standoff_corefs)