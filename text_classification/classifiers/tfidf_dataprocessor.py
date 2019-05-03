from commons.util_methods import iterable_to_batches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
import re


def identity_dummy_method(x):
    '''
    just to fool the scikit-learn vectorizer
    '''
    return x

def get_nrams(string, min_n=3, max_n=5):
    return [string[k:k + ngs] for ngs in range(min_n, max_n + 1) for k in range(len(string) - ngs)]

def regex_tokenizer(text, pattern=r"(?u)\b\w\w+\b"):# pattern stolen from scikit-learn
    return [m.group() for m in re.finditer(pattern, text)]

def text_to_bow(text):
    return regex_tokenizer(text)

def windowed_bow(data):
    raw_bows = [text_to_bow(d['utterance'])+['SPEAKER__'+d['speaker']] for d in data]

    def window_prefixed_bows(idx,before=-8,after=2):
        return [str(i)+'__'+tok
                for i in range(before,after)
                if idx+i<len(data) and idx+i>=0
                if data[idx+i]['debatefile']==data[idx]['debatefile']
                for tok in raw_bows[idx+i]]

    prefixed_bows = [window_prefixed_bows(idx) for idx in range(len(data))]
    return prefixed_bows

def raw_bow(texts):
    raw_bows = [text_to_bow(text) for text in texts]
    return raw_bows

class TfIdfTextClfDataProcessor(object):

    def __init__(self,
                 text_to_bow_fun,
                 ) -> None:
        super().__init__()
        self.text_to_bow_fun=text_to_bow_fun
        self.target_binarizer = MultiLabelBinarizer()

        self.vectorizer = TfidfVectorizer(sublinear_tf=True,
                                     preprocessor=identity_dummy_method,
                                     tokenizer=identity_dummy_method,
                                     ngram_range=(1, 1),
                                     max_df=0.75, min_df=2,
                                     max_features=30000,
                                     stop_words=None  # 'english'
                                     )
    def fit(self,data):

        self.vectorizer.fit(self.text_to_bow_fun([d['text'] for d in data]))
        self.target_binarizer.fit([d['labels'] for d in data])


    def process_inputs_and_targets(self,data):
        inputs = self.process_inputs(data)
        targets = self.target_binarizer.transform([d['labels'] for d in data]).astype('float32')
        return inputs,targets

    def process_inputs(self, data):
        bow = self.text_to_bow_fun([d['text'] for d in data])
        csr = self.vectorizer.transform(bow)
        return csr

