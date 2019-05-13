import sys
sys.path.append('.')
from scipy.sparse import csr_matrix #TODO(tilo): if not imported before torch it throws: ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found

from pytorch_util.pytorch_DataLoaders import GetBatchFunDatasetWrapper
from pytorch_util import pytorch_methods
from sklearn.externals import joblib
from model_evaluation.classification_metrics import calc_classification_metrics
import json
import os
import time
from collections import Counter
from pprint import pprint
from typing import NamedTuple
import torch.nn.functional as F

import torch
from commons import data_io
from commons.util_methods import iterable_to_batches
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn as nn
from torch.utils.data import Dataset

from pytorch_util.multiprocessing_proved_dataloading import build_messaging_DataLoader_from_dataset_builder
from pytorch_util.pytorch_methods import get_device, iterate_and_time
from pytorchic_bert import optim
from pytorchic_bert import tokenization
from pytorchic_bert.preprocessing import Pipeline, SentencePairTokenizer, AddSpecialTokensWithTruncation, TokenIndexing
from pytorchic_bert.utils import set_seeds
from text_classification.classifiers.common import GenericClassifier
import pytorchic_bert.selfattention_encoder as selfatt_enc


class TrainConfig(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 42 # random seed
    batch_size: int = 32
    lr: int = 2e-5 # learning rate
    n_epochs: int = 10 # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup: float = 0.1
    total_steps: int = batch_size*n_epochs # total number of steps to train

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))


class AttentionClassifier(nn.Module):

    def __init__(self, cfg: selfatt_enc.BertConfig, n_labels):
        super().__init__()
        self.bert_encoder = selfatt_enc.EncoderStack(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.classifier = nn.Linear(cfg.dim, n_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        h = self.bert_encoder(input_ids, segment_ids, input_mask)
        # only use the first h in the sequence
        pooled_h = torch.tanh(self.fc(h[:, 0]))
        logits = self.classifier(self.drop(pooled_h))
        return logits

    @staticmethod
    def build_loss_fun(criterion):
        def loss_fun(model,batch):
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model.forward(input_ids, segment_ids, input_mask)
            return criterion(logits, label_id)
        return loss_fun

class DataProcessor(object):

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

class AttentionClassifierPytorch(GenericClassifier):
    def __init__(self,
                 train_config:TrainConfig,
                 model_cfg: selfatt_enc.BertConfig,
                 dataprocessor:DataProcessor,
                 ) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.train_config = train_config
        self.dataprocessor =  dataprocessor
        self.device = get_device()
        self.model = None

    def _build_model(self):
        return AttentionClassifier(self.model_cfg, len(self.dataprocessor.target_binarizer.classes_.tolist()))

    def fit(self,X,y=None,
            save_dir='./save_dir',
            model_file = None,
            pretrain_file = None,
            data_parallel = True
            ):
        self.save_dir = save_dir
        self.dataprocessor.fit(X)

        dataloader = build_messaging_DataLoader_from_dataset_builder(
            dataset_builder=lambda _:GetBatchFunDatasetWrapper(self.dataprocessor.build_get_batch_fun(X,batch_size=32)),
            message_supplier=lambda :None,
            collate_fn=lambda x:[t.to(self.device) for t in x[0]],
            num_workers=0
        )

        model,loss_fun = self.prepare_model_for_training(data_parallel, model_file, pretrain_file)

        self.optimizer = optim.optim4GPU(self.train_config, self.model) #TODO(tilo):holyJohn!
        # if False:
        #     p_cond = lambda p_name,p: 'bert_encoder' not in p_name and p.requires_grad
        # else:
        #     p_cond = lambda p_name,p: p.requires_grad
        # params = [p for p_name, p in self.model.named_parameters() if p_cond(p_name,p)]
        #
        # self.optimizer = torch.optim.RMSprop(params, lr=0.01)

        def train_on_batch(batch):
            self.optimizer.zero_grad()
            loss = loss_fun(model,batch).mean()  # mean() for Data Parallelism
            loss.backward()
            self.optimizer.step()
            return loss.item()

        pytorch_methods.train(train_on_batch, dataloader, self.train_config.n_epochs,verbose=True)
        return self

    def prepare_model_for_training(self, data_parallel, model_file, pretrain_file):
        if self.model is None:
            self.model = self._build_model()

        if model_file is not None: # to resume training
            raise NotImplementedError
        elif pretrain_file is not None:
            self.load_bert_encoder(pretrain_file)

        loss_fun = self.model.build_loss_fun(nn.CrossEntropyLoss())
        self.model.train()
        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)
        return model,loss_fun

    def predict_proba(self,X,data_parallel=True):

        data_iter = build_messaging_DataLoader_from_dataset_builder(
            dataset_builder=lambda _: GetBatchFunDatasetWrapper(self.dataprocessor.build_get_batch_fun(X, batch_size=1024)),
            message_supplier=lambda :None,
            collate_fn=lambda x:x[0],
            num_workers=0
        )

        self.model.eval()
        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)

        with torch.no_grad():
            probas = [p for batch in data_iter
                    for p in F.softmax(model(*batch[:-1]), dim=1).data.cpu().numpy().tolist()]

            results= np.array(probas)
        return results.astype('float64')

    def predict_proba_encode_targets(self, data):
        probas = self.predict_proba(data)
        targets = self.target_binarizer.transform([d['labels'] for d in data]).astype('int64')
        return probas, targets

    @property
    def target_binarizer(self):
        return self.dataprocessor.target_binarizer

    def load_bert_encoder(self,model_file):
        assert os.path.isfile(model_file)
        loaded = torch.load(model_file, map_location=None if torch.cuda.is_available() else 'cpu')
        self.model.bert_encoder.load_state_dict(loaded)

        # if pretrain_file.endswith('.ckpt'):  # checkpoint file in tensorflow
        #     assert False
        #     # checkpoint.load_model(self.model.transformer, pretrain_file)
        # elif pretrain_file.endswith('.pt'):  # pretrain model file in pytorch
        #     self.model.bert_encoder.load_state_dict(
        #         {key[12:]: value
        #          for key, value in torch.load(pretrain_file).items()
        #          if key.startswith('transformer')}
        #     )  # load only transformer parts
        # else:
        #     assert False

    @staticmethod
    def load(path):
        obj:AttentionClassifierPytorch = joblib.load(path + '/bertattClf.pkl')
        obj.model = obj._build_model()
        obj.model.load_state_dict(torch.load(path+'/bert_finetuned_clf_module.pt',map_location=None if torch.cuda.is_available() else 'cpu'))
        return obj
        # if model_file:
        #     print('Loading the model from', model_file)
        #     self.model.load_state_dict(torch.load(model_file,map_location=None if torch.cuda.is_available() else 'cpu'))
        #
        # elif pretrain_file: # use pretrained transformer
        #     print('Loading the pretrained model from', pretrain_file)

    def save(self, path):
        self.model.cpu()
        torch.save(self.model.state_dict(), path + '/bert_finetuned_clf_module.pt')
        self.model = None
        # joblib.dump({k:v for k,v in self.__dict__.items() if k not in ['model','optimizer']}, save_dir+'/bertattClf.pkl')
        joblib.dump(self, path + '/bertattClf.pkl')




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
    dp = DataProcessor(vocab_file=vocab,max_len=max_len)
    pipeline = AttentionClassifierPytorch(cfg, model_cfg,dp)
    pretrain_file = home + '/data/models/uncased_L-12_H-768_A-12/bert_encoder_pytorch.pt'
    # pretrain_file = None
    save_path = home + '/nlp/saved_models'
    pipeline.fit(train_data, pretrain_file=pretrain_file)
    pipeline.save(save_path)
    del pipeline

    pipeline = AttentionClassifierPytorch.load(save_path)
    evaluate(pipeline,train_data,test_data)
