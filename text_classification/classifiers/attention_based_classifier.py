import json
import os
from abc import abstractmethod
from typing import NamedTuple

import torch
import torch.nn.functional as F
from sklearn.externals import joblib
from torch import nn as nn

import pytorchic_bert.selfattention_encoder as selfatt_enc
from pytorch_util import pytorch_methods
from pytorch_util.multiprocessing_proved_dataloading import build_messaging_DataLoader_from_dataset_builder
from pytorch_util.pytorch_DataLoaders import GetBatchFunDatasetWrapper
from pytorch_util.pytorch_methods import get_device
from pytorchic_bert import optim
from text_classification.classifiers.common import GenericClassifier
import numpy as np

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

    @abstractmethod
    def fit(self, data):
        raise NotImplementedError

    @abstractmethod
    def transform(self,data):
        raise NotImplementedError

    @abstractmethod
    def build_get_batch_fun(self,raw_data,batch_size):
        raise NotImplementedError

class AttentionClassifierPytorch(GenericClassifier):
    def __init__(self,
                 train_config:TrainConfig,
                 model_cfg: selfatt_enc.BertConfig,
                 dataprocessor: DataProcessor,
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



