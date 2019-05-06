import sys
sys.path.append('.')
from scipy.sparse import csr_matrix
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
from pytorchic_bert import checkpoint, optim
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
        self.transformer = selfatt_enc.EncoderStack(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.classifier = nn.Linear(cfg.dim, n_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        h = self.transformer(input_ids, segment_ids, input_mask)
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
                 class_labels,
                 max_len,
                 ):
        super().__init__()
        self.target_binarizer = MultiLabelBinarizer()

        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
        self.pipeline = Pipeline([SentencePairTokenizer(tokenizer.convert_to_unicode, tokenizer.tokenize),
                             AddSpecialTokensWithTruncation(max_len),
                             TokenIndexing(tokenizer.convert_tokens_to_ids, class_labels, max_len)
                             ]
                            )
    def fit(self, data):
        self.target_binarizer.fit([d['labels'] for d in data])

    def transform(self,data):
        return [self.pipeline.transform((d['labels'],d['text'],d['textb'])) for d in data]



class AttentionClassifierPytorch(GenericClassifier):
    def __init__(self,
                 cfg,
                 model_cfg,
                 vocab_file,
                 max_len,
                 class_labels
                 ) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.cfg = cfg
        self.vocab_file = vocab_file
        self.max_len = max_len
        self.class_labels = class_labels
        self.num_labels = len(class_labels)

    def fit(self,X,y=None,
            save_dir='./save_dir',
            model_file = None,
            pretrain_file = None,
            data_parallel = True
            ):
        self.save_dir = save_dir
        self.dataprocessor = DataProcessor(vocab_file=self.vocab_file,class_labels=self.class_labels,max_len=self.max_len)
        self.dataprocessor.fit(X)

        dataloader = build_messaging_DataLoader_from_dataset_builder(
            dataset_builder=lambda i:AttentionDataset(X,self.dataprocessor,batch_size=32),
            message_supplier=lambda :None,
            collate_fn=lambda x:x[0],
            num_workers=0
        )
        self.model = AttentionClassifier(self.model_cfg, self.num_labels)
        self.optimizer = optim.optim4GPU(self.cfg, self.model) #TODO(tilo):holyJohn!
        # self.optimizer = torch.optim.RMSprop([p for p in self.model.parameters() if p.requires_grad], lr=0.01)
        self.train(dataloader,model_file, pretrain_file, data_parallel)
        return self

    def predict_proba(self,X):

        def pred_batch_fun(model, batch):
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            return logits

        data_iter = build_messaging_DataLoader_from_dataset_builder(
            dataset_builder=lambda i:AttentionDataset(X,self.dataprocessor,batch_size=128),
            message_supplier=lambda :None,
            collate_fn=lambda x:x[0],
            num_workers=0
        )

        batch_results = self.eval(data_iter,pred_batch_fun,model_file=None)
        results= torch.cat(batch_results)
        return results.cpu().numpy().astype('float64')

    def predict_proba_encode_targets(self, data):
        probas = self.predict_proba(data)
        targets = self.target_binarizer.transform([d['labels'] for d in data]).astype('int64')
        return probas, targets

    def evaluate(self,X):
        def calc_acc(model, batch):
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            _, label_pred = logits.max(1)
            result = (label_pred == label_id).float() #.cpu().numpy()
            return result

        data_iter = build_messaging_DataLoader_from_dataset_builder(
            dataset_builder=lambda i:AttentionDataset(X,self.dataprocessor,batch_size=128),
            message_supplier=lambda :None,
            collate_fn=lambda x:x[0],
            num_workers=0
        )

        batch_results = self.eval(data_iter,calc_acc,model_file=None)
        results= torch.cat(batch_results).mean()
        return results.cpu().numpy().astype('float64')

    def train(self, dataloader, model_file=None, pretrain_file=None, data_parallel=True):
        self.device = get_device()
        self.model.train()
        self.load(model_file, pretrain_file)
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


    def eval(self, data_iter, evaluate, model_file, data_parallel=True):

        self.model.eval() # evaluation mode
        # self.load(model_file, None)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        results = [] # prediction results
        # iter_bar = tqdm(data_iter, desc='Iter (loss=X.XXX)')
        iter_bar = data_iter
        with torch.no_grad():  # evaluation without gradient calculation
            for batch in iter_bar:
                batch = [t.to(self.device) for t in batch]
                result = evaluate(model, batch) # accuracy to print
                results.append(result)

        return results

    @property
    def target_binarizer(self):
        return self.dataprocessor.target_binarizer


    def load(self, model_file, pretrain_file):
        if model_file:
            print('Loading the model from', model_file)
            self.model.load_state_dict(torch.load(model_file))

        elif pretrain_file: # use pretrained transformer
            print('Loading the pretrained model from', pretrain_file)
            if pretrain_file.endswith('.ckpt'): # checkpoint file in tensorflow
                checkpoint.load_model(self.model.transformer, pretrain_file)
            elif pretrain_file.endswith('.pt'): # pretrain model file in pytorch
                self.model.transformer.load_state_dict(
                    {key[12:]: value
                        for key, value in torch.load(pretrain_file).items()
                        if key.startswith('transformer')}
                ) # load only transformer parts
            else:
                assert False

    def save(self, i):
        assert False
        torch.save(self.model.state_dict(), # save model object before nn.DataParallel
            os.path.join(self.save_dir, 'model_steps_'+str(i)+'.pt'))

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

    from pathlib import Path
    home = str(Path.home())
    cwd = home
    data = get_data(home + '/data/glue/MRPC/train.tsv')

    label_counter = Counter([d['labels'] for d in data])
    pprint(label_counter)

    train_data = data
    test_data = get_data(home + '/data/glue/MRPC/dev.tsv')
    label_counter = Counter([d['labels'] for d in test_data])
    print(label_counter)

    cfg = AttentionClassifierConfig.from_json('pytorchic_bert/config/train_mrpc.json')

    model_cfg = selfatt_enc.BertConfig.from_json('pytorchic_bert/config/bert_base.json')
    max_len = 128

    set_seeds(cfg.seed)

    vocab = cwd+'/data/models/uncased_L-12_H-768_A-12/vocab.txt'

    pipeline = AttentionClassifierPytorch(cfg, model_cfg, vocab, max_len, list(label_counter.keys()))
    pretrain_file = home + '/data/models/uncased_L-12_H-768_A-12/bert_model.ckpt'
    # pretrain_file = None
    pipeline.fit(data, pretrain_file=pretrain_file)


    print(pipeline.evaluate(train_data))
    print(pipeline.evaluate(test_data))
    # proba, y_train = pipeline.predict_proba_encode_targets(train_data)
    # pred = np.array(proba == np.expand_dims(np.max(proba, axis=1), 1), dtype='int64')
    # target_names = pipeline.target_binarizer.classes_.tolist()
    # train_scores = calc_classification_metrics(proba, pred, y_train, target_names=target_names)
    # pprint(train_scores['f1-micro'])
    # proba, y_test = pipeline.predict_proba_encode_targets(test_data)
    # pred = np.array(proba == np.expand_dims(np.max(proba, axis=1), 1), dtype='int64')
    # test_scores = calc_classification_metrics(proba, pred, y_test, target_names=target_names)
    # pprint(test_scores['f1-micro'])
