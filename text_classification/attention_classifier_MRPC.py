
import sys
sys.path.append('.')
from scipy.sparse import csr_matrix #TODO(tilo): if not imported before torch it throws: ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found

from text_classification.classifiers.attention_based_classifier import TrainConfig, AttentionClassifierPytorch, \
    DataProcessor
from commons.util_methods import iterable_to_batches
from sklearn.preprocessing import MultiLabelBinarizer

from pytorchic_bert import tokenization
from pytorchic_bert.preprocessing import Pipeline, SentencePairTokenizer, AddSpecialTokensWithTruncation, TokenIndexing

from model_evaluation.classification_metrics import calc_classification_metrics
import time
from collections import Counter
from pprint import pprint

import torch
from commons import data_io

from pytorchic_bert.utils import set_seeds
import pytorchic_bert.selfattention_encoder as selfatt_enc

class TwoSentDataProcessor(DataProcessor):

    def __init__(self,
                 vocab_file,
                 max_len,
                 ):
        super().__init__()
        self.vocab_file = vocab_file
        self.max_len = max_len
        self.target_binarizer = MultiLabelBinarizer()

    def fit(self, data):
        self.target_binarizer.fit([d['labels'] for d in data])
        class_labels = self.target_binarizer.classes_.tolist()
        tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=True)
        self.pipeline = Pipeline([SentencePairTokenizer(tokenizer.convert_to_unicode, tokenizer.tokenize),
                             AddSpecialTokensWithTruncation(self.max_len),
                             TokenIndexing(tokenizer.convert_tokens_to_ids, class_labels, self.max_len)
                             ]
                            )

    def transform(self,data):
        return [self.pipeline.transform((d['labels'], d['text'], d['textb'])) for d in data]

    def build_get_batch_fun(self,raw_data,batch_size):

        tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*self.transform(raw_data))]

        def build_idx_batch_generator(batch_size):
            # self.tensors = [tensor[torch.randperm(tensor.shape[0])] for tensor in self.tensors]
            # idx = torch.randperm(self.tensors[0].shape[0])
            idx = range(tensors[0].shape[0])
            return iterable_to_batches(iter(idx), batch_size)

        batch_indizes_g =[0]
        batch_indizes_g[0] = build_idx_batch_generator(batch_size)

        def get_batch(message):
            try:
                batch = next(batch_indizes_g[0])
            except StopIteration:
                # self.tensors = [tensor[torch.randperm(tensor.shape[0])] for tensor in self.tensors]
                batch_indizes_g[0] = build_idx_batch_generator(batch_size)
                raise StopIteration

            return tuple(tensor[batch] for tensor in tensors)
        return get_batch


if __name__ == '__main__':
    import numpy as np

    def get_data(file):
        def parse_line(line):
            label, id1, id2, texta, textb = line.split('\t')
            return {
                'text': texta,
                'textb': textb,
                'labels': label}

        lines_g = data_io.read_lines(file)
        next(lines_g)
        data = [parse_line(line) for line in lines_g]
        return data

    def evaluate(pipeline1, train_data, test_data):
        target_names = pipeline1.target_binarizer.classes_.tolist()

        proba, y_train = pipeline1.predict_proba_encode_targets(train_data)
        pred = np.array(proba == np.expand_dims(np.max(proba, axis=1), 1), dtype='int64')
        train_scores = calc_classification_metrics(proba, pred, y_train, target_names=target_names)
        pprint('Train-f1-micro: %0.2f' % train_scores['f1-micro'])

        proba, y_test = pipeline1.predict_proba_encode_targets(test_data)
        pred = np.array(proba == np.expand_dims(np.max(proba, axis=1), 1), dtype='int64')
        test_scores = calc_classification_metrics(proba, pred, y_test, target_names=target_names)
        pprint('Test-f1-micro: %0.2f' % test_scores['f1-micro'])

        print('evaluating took: %0.2f secs' % (time.time() - start))


    def load_data():
        train_data = get_data(home + '/data/glue/MRPC/train.tsv')
        label_counter = Counter([d['labels'] for d in train_data])
        pprint(label_counter)
        test_data = get_data(home + '/data/glue/MRPC/dev.tsv')
        label_counter = Counter([d['labels'] for d in test_data])
        print(label_counter)
        return train_data, test_data

    start = time.time()
    from pathlib import Path
    home = str(Path.home())
    train_data, test_data = load_data()

    cfg = TrainConfig(seed=42,n_epochs=1)#.from_json('pytorchic_bert/config/train_mrpc.json')
    model_cfg = selfatt_enc.BertConfig.from_json('pytorchic_bert/config/bert_base.json')
    max_len = 128

    set_seeds(cfg.seed)

    vocab = home+'/data/models/uncased_L-12_H-768_A-12/vocab.txt'
    dp = TwoSentDataProcessor(vocab_file=vocab, max_len=max_len)
    pipeline = AttentionClassifierPytorch(cfg, model_cfg,dp)
    pretrain_file = home + '/data/models/uncased_L-12_H-768_A-12/bert_encoder_pytorch.pt'
    # pretrain_file = None
    save_path = home + '/nlp/saved_models'
    pipeline.fit(train_data, pretrain_file=pretrain_file)
    pipeline.save(save_path)
    del pipeline

    pipeline = AttentionClassifierPytorch.load(save_path)
    evaluate(pipeline,train_data,test_data)
