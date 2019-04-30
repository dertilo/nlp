import re
from collections import Counter
from typing import Tuple, Iterable, List, Callable

UNKNOWN='<UNKNOWN>'
PADDING='<PADDING>'

def regex_tokenizer(text, pattern=r"(?u)\b\w\w+\b")->List[Tuple[int,int,str]]:# pattern stolen from scikit-learn
    return [(m.start(),m.end(),m.group()) for m in re.finditer(pattern, text)]

class TokenizeIndexer(object):
    def __init__(self,tokenizer_fun=regex_tokenizer,min_freq=1):
        # self.vocab_size = vocab_size
        self.tokenizer_fun:Callable[[str],List[Tuple]] = tokenizer_fun
        self.token2key = {}
        self.min_freq = min_freq
        self.key2token={}
        self.key2freq={}

    def fit(self, texts_g:Iterable):
        token_generator = (token for text in texts_g for start, end, token in self.tokenizer_fun(text))
        _word_counter = Counter(iter(token_generator))
        # min_freq_tokens = ((token,freq) for token,freq in self._word_counter.items() if freq>=self.min_freq )
        for token,freq in _word_counter.items():
            assert isinstance(token,str)
            if freq >= self.min_freq:
                self.add_token(token,freq)
        self.add_token(UNKNOWN)
        self.add_token(PADDING)


    def add_token(self,token,freq=1):
        idx = len(self.token2key)
        if token not in self.token2key:
            self.token2key[token] = idx
            self.key2token[idx]=token
            self.key2freq[idx]=freq
        else:
            self.key2freq[self.token2key[token]]+=freq

    def transform_to_token_spans(self,texts:Iterable):
        for text in texts:
            yield [(start,end,self.token2key.get(token, self.token2key[UNKNOWN])) for start, end, token in self.tokenizer_fun(text)]

    def transform_to_seq(self, text)->List[int]:
            return [self.token2key.get(token, self.token2key[UNKNOWN]) for start, end, token in self.tokenizer_fun(text)]

    def get_vocab_size(self):
        return len(self.key2token)

    def get_token2freq(self):
        return {token:self.key2freq[key] for key,token in self.key2token.items()}