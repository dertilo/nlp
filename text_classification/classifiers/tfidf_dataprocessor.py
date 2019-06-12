import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer


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

class TfIdfTextClfDataProcessor(object):

    def __init__(self,
                 process_data_to_bows_fun,
                 get_targets_fun
                 ) -> None:
        super().__init__()
        if get_targets_fun is None:
            get_targets_fun = lambda x:x['labels']

        self.get_targets_fun = get_targets_fun
        self.process_data_to_bows_fun=process_data_to_bows_fun
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

        self.vectorizer.fit(self.process_data_to_bows_fun(data))
        self.target_binarizer.fit([self.get_targets_fun(d) for d in data])


    def process_inputs_and_targets(self,data):
        inputs = self.process_inputs(data)
        targets = self.target_binarizer.transform([self.get_targets_fun(d) for d in data]).astype('float32')
        return inputs,targets

    def process_inputs(self, data):
        bow = self.process_data_to_bows_fun(data)
        csr = self.vectorizer.transform(bow)
        return csr

