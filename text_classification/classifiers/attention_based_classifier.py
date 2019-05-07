import sys
sys.path.append('.')

from sklearn.externals import joblib
from scipy.sparse import csr_matrix
from model_evaluation.classification_metrics import calc_classification_metrics
import json
import os
import time
from collections import Counter
from pprint import pprint
from typing import NamedTuple

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


class AttentionClassifierConfig(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    batch_size: int = 32
    lr: int = 5e-5 # learning rate
    n_epochs: int = 10 # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup: float = 0.1
    save_steps: int = 100 # interval for saving model
    total_steps: int = 100000 # total number of steps to train

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))


class AttentionClassifier(nn.Module):

    def __init__(self, cfg, n_labels):
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


class AttentionDataset(Dataset):

    def __init__(self,
                 raw_data,
                 dataprocessor,
                 batch_size
                 ):

        self.dataprocessor = dataprocessor
        self.batch_size = batch_size
        data = self.dataprocessor.transform(raw_data)
        self.tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]
        self.batch_indizes_g = self.build_idx_batch_generator(batch_size)

    def build_idx_batch_generator(self, batch_size):
        # self.tensors = [tensor[torch.randperm(tensor.shape[0])] for tensor in self.tensors]
        # idx = torch.randperm(self.tensors[0].shape[0])
        idx = range(self.tensors[0].shape[0])
        return iterable_to_batches(iter(idx), batch_size)

    def __getitem__(self, index):
        message = index
        try:
            batch = next(self.batch_indizes_g)
        except StopIteration:
            # self.tensors = [tensor[torch.randperm(tensor.shape[0])] for tensor in self.tensors]
            self.batch_indizes_g = self.build_idx_batch_generator(self.batch_size)
            raise StopIteration

        return tuple(tensor[batch] for tensor in self.tensors)

    def __len__(self):
        return int(np.ceil(self.tensors[0].shape[0]/self.batch_size))


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
        processed_data = [self.pipeline.transform((d['labels'], d['text'], d['textb'])) for d in data]
        # processed_data = [(input_ids, segment_ids, input_mask, self.target_binarizer.classes_.tolist().index(d['labels'][0]))
        #                   for d,(input_ids, segment_ids, input_mask, label_id) in zip(data,processed_data)]
        return processed_data



class AttentionClassifierPytorch(GenericClassifier):
    def __init__(self,
                 cfg,
                 model_cfg,
                 vocab_file,
                 max_len,
                 ) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.cfg = cfg
        self.dataprocessor = DataProcessor(vocab_file=vocab_file,max_len=max_len)
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
        if self.model is None:
            self.model = self._build_model()
        self.save_dir = save_dir
        self.dataprocessor.fit(X)

        dataloader = build_messaging_DataLoader_from_dataset_builder(
            dataset_builder=lambda i:AttentionDataset(X,self.dataprocessor,batch_size=32),
            message_supplier=lambda :None,
            collate_fn=lambda x:x[0],
            num_workers=0
        )
        self.optimizer = optim.optim4GPU(self.cfg, self.model) #TODO(tilo):holyJohn!
        # self.optimizer = torch.optim.RMSprop([p for p in self.model.parameters() if p.requires_grad], lr=0.01)
        self.train(dataloader,model_file, pretrain_file, data_parallel)
        return self

    def predict_proba(self,X,data_parallel=True):

        data_iter = build_messaging_DataLoader_from_dataset_builder(
            dataset_builder=lambda i:AttentionDataset(X,self.dataprocessor,batch_size=1024),
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

    def train(self, dataloader, model_file=None, pretrain_file=None, data_parallel=True):
        self.model.train()
        if model_file is not None:
            raise NotImplementedError
        elif pretrain_file is not None:
            self.load_bert_encoder(pretrain_file)

        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)

        criterion = nn.CrossEntropyLoss()

        global_step = 0 # global iteration steps regardless of epochs

        def train_on_batch(batch):
            batch = [t.to(self.device) for t in batch]
            self.optimizer.zero_grad()

            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            loss = criterion(logits, label_id).mean()  # mean() for Data Parallelism
            loss.backward()
            self.optimizer.step()
            return loss.item()

        for e in range(self.cfg.n_epochs):
            loss_sum = 0. # the sum of iteration losses to get average loss in every epoch
            duration_sum = 0

            start = time.time()
            iter_bar = iter(dataloader)
            for i, (batch,dur) in enumerate(iterate_and_time(iter_bar)):
                loss = train_on_batch(batch)

                global_step += 1
                loss_sum += loss
                duration_sum+=dur

            print('Epoch %d/%d : Average Loss %5.3f;epoch-dur: %0.1f secs; dataloader-dur: %0.1f secs'
                  %(e+1, self.cfg.n_epochs, loss_sum/(i+1),time.time()-start,duration_sum))
        # self.save(global_step)

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

    def save(self,save_dir):
        self.model.cpu()
        print(self.__dict__.keys())
        torch.save(self.model.state_dict(), save_dir+'/bert_finetuned_clf_module.pt')
        self.model = None
        # joblib.dump({k:v for k,v in self.__dict__.items() if k not in ['model','optimizer']}, save_dir+'/bertattClf.pkl')
        joblib.dump(self, save_dir+'/bertattClf.pkl')


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

if __name__ == '__main__':
    start = time.time()
    from pathlib import Path
    home = str(Path.home())
    cwd = home
    train_data = get_data(home + '/data/glue/MRPC/train.tsv')
    label_counter = Counter([d['labels'] for d in train_data])
    pprint(label_counter)

    test_data = get_data(home + '/data/glue/MRPC/dev.tsv')
    label_counter = Counter([d['labels'] for d in test_data])
    print(label_counter)

    cfg = AttentionClassifierConfig.from_json('pytorchic_bert/config/train_mrpc.json')
    model_cfg = selfatt_enc.BertConfig.from_json('pytorchic_bert/config/bert_base.json')
    max_len = 128

    set_seeds(cfg.seed)

    vocab = cwd+'/data/models/uncased_L-12_H-768_A-12/vocab.txt'

    pipeline = AttentionClassifierPytorch(cfg, model_cfg, vocab, max_len)
    pretrain_file = home + '/data/models/uncased_L-12_H-768_A-12/bert_encoder_pytorch.pt'
    # pretrain_file = None
    save_path = home + '/nlp/saved_models'
    # save_path = '/tmp/saved_models'
    # pipeline.fit(train_data, pretrain_file=pretrain_file)
    # print('before-save: %0.2f'%pipeline.evaluate(test_data))
    # pipeline.save(save_path)
    pipeline=None
    # trained_model = '/code/NLP/nlp/saved_models/model_steps_345.pt'
    # trained_model = '/nlp/saved_models/model_steps_345.pt'
    # pipeline.load(train_data, model_file=home + trained_model, pretrain_file=None)
    # print('loading took: %0.2f'%(time.time()-start))
    start = time.time()

    pipeline1 = AttentionClassifierPytorch.load(save_path)
    pipeline1.target_binarizer.classes_ = pipeline1.target_binarizer.classes_[[1,0]]
    target_names = pipeline1.target_binarizer.classes_.tolist()
    # proba, y_train = pipeline.predict_proba_encode_targets(train_data)
    # pred = np.array(proba == np.expand_dims(np.max(proba, axis=1), 1), dtype='int64')
    # train_scores = calc_classification_metrics(proba, pred, y_train, target_names=target_names)
    # pprint(train_scores['f1-micro'])
    proba, y_test = pipeline1.predict_proba_encode_targets(test_data)
    pred = np.array(proba == np.expand_dims(np.max(proba, axis=1), 1), dtype='int64')
    print(pred.sum(axis=0))
    test_scores = calc_classification_metrics(proba, pred, y_test, target_names=target_names)
    pprint('f1-micro: %0.2f'%test_scores['f1-micro'])
    pprint('acc: %0.2f'%test_scores['accuracy'])
    print('evaluating took: %0.2f'%(time.time()-start))
