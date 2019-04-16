from typing import Tuple, List

from commons import data_io
from sklearn.datasets import fetch_20newsgroups
import numpy as np

def get_20newsgroups_data(train_test,
                          categories=None,
                          truncate_to:int=None,
                          min_num_tokens=0,
                          random_state=42)->List[Tuple[str,str]]:
    """
     'alt.atheism',
     'comp.graphics',
     'comp.os.ms-windows.misc',
     'comp.sys.ibm.pc.hardware',
     'comp.sys.mac.hardware',
     'comp.windows.x',
     'misc.forsale',
     'rec.autos',
     'rec.motorcycles',
     'rec.sport.baseball',
     'rec.sport.hockey',
     'sci.crypt',
     'sci.electronics',
     'sci.med',
     'sci.space',
     'soc.religion.christian',
     'talk.politics.guns',
     'talk.politics.mideast',
     'talk.politics.misc',
     'talk.religion.misc'
    """
    data = fetch_20newsgroups(subset=train_test,
                              shuffle=True,
                              remove=('headers', 'footers', 'quotes'),
                              categories=categories,
                              random_state=random_state
                              )
    target_names = data.target_names


    data = [(d, target_names[target]) for d, target in zip(data.data, data.target) if len(d.split(' ')) > min_num_tokens]
    if truncate_to is not None:
        def truncate(text):
            return text[0:min(len(text), truncate_to)]
        data = [(truncate(d),t) for d,t in data]
    return data


def get_GermEval2017_TaskB_data(data_file='some-path/GermEval2017/train-2017-09-15.tsv',
                                limit = np.Inf):
    '''
    download data from: https://sites.google.com/view/germeval2017-absa/data
    '''

    def process_line(line):
        split = line.split('\t')
        url = split[0];
        text = split[1];
        relevance = split[2];
        sentiment = split[3];
        datum = {'text': text, 'relevance': relevance, 'sentiment': sentiment}
        if len(split)>4:
            aspect_Polarity = split[4]
            datum['aspect_polarity']=aspect_Polarity
        return datum

    data = []
    for line in data_io.read_lines(data_file, limit=limit):
        try:
            datum = process_line(line)
        except:
            print(line)
            continue
        data.append(datum)

    return data