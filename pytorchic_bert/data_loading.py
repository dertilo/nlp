import csv
import itertools
import json
from typing import List

import torch
from torch.utils.data import Dataset

from pytorchic_bert import tokenization
from pytorchic_bert.preprocessing import Pipeline, SentencePairTokenizer, AddSpecialTokensWithTruncation, TokenIndexing


class CsvDataset(Dataset):

    labels = None
    def __init__(self, file, pipeline:Pipeline): # cvs file and pipeline object
        Dataset.__init__(self)

        with open(file, "r") as f:
            lines = csv.reader(f, delimiter='\t', quotechar=None)
            data = [pipeline.transform(instance) for instance in self.get_instances(lines)]

        self.tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def get_instances(self, lines):
        """ get instance array from (csv-separated) line list """
        raise NotImplementedError

class TextLabelTupleDataset(Dataset):

    def __init__(self, raw_data:List[dict], pipeline:Pipeline) -> None:
        super().__init__()
        data = [pipeline.transform((d['label'],d['text'],'')) for d in raw_data]
        self.tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)


class MRPC(CsvDataset):
    """ Dataset class for MRPC """
    labels = ("0", "1") # label names
    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None): # skip header
            yield line[0], line[3], line[4] # label, text_a, text_b


class MNLI(CsvDataset):
    """ Dataset class for MNLI """
    labels = ("contradiction", "entailment", "neutral") # label names
    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None): # skip header

            yield line[-1], line[8], line[9] # label, text_a, text_b


if __name__ == '__main__':
    vocab = '/home/tilo/data/models/uncased_L-12_H-768_A-12/vocab.txt'
    max_len = 128
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    TaskDataset = TextLabelTupleDataset # task dataset class according to the task

    pipeline = Pipeline([SentencePairTokenizer(tokenizer.convert_to_unicode, tokenizer.tokenize),
                         AddSpecialTokensWithTruncation(max_len),
                         TokenIndexing(tokenizer.convert_tokens_to_ids, TaskDataset.labels, max_len)
                         ]
                        )
    data_file = '/home/tilo/data/fact_checking/clef2019.jsonl'
    dataset = TaskDataset(data_file, pipeline)
    d = dataset[0]
    print(d)