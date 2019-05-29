import os

from commons import data_io

'''
git clone https://github.com/apepa/clef2019-factchecking-task1.git
train-data is in: clef2019-factchecking-task1/data/training
'''

def parse_line(filename, line):
    line_id, speaker, utterance, label = line.split('\t')
    return {
        'debatefile':filename,
        'line_id':line_id,
        'speaker':speaker,
        'utterance':utterance,
        'label': label.replace('\r','')
    }

def get_Clef2019_data(data_path):
    return [parse_line(f,l) for f in os.listdir(data_path) for l in data_io.read_lines(os.path.join(data_path,f))]
