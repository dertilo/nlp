import json
import os
from typing import NamedTuple

import torch
import torch.nn.functional as F
from sklearn.externals import joblib
from torch import nn as nn

import pytorchic_bert.selfattention_encoder as selfatt_enc
from pytorch_util import pytorch_methods
from pytorch_util.multiprocessing_proved_dataloading import build_messaging_DataLoader_from_dataset_builder
from pytorch_util.pytorch_DataLoaders import GetBatchFunDatasetWrapper
from pytorch_util.pytorch_methods import get_device, to_torch_to_cuda
from pytorchic_bert import optim
from text_classification.classifiers.common import GenericClassifier, DataProcessorInterface
import numpy as np

class TrainConfig(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 42 # random seed
    batch_size: int = 32
    lr: float = 2e-5 # learning rate
    n_epochs: int = 10 # the number of epoch
    tol:float = 0
    patience:int = 3
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
            target = batch.pop('target')
            logits = model.forward(**batch)
            return criterion(logits, target)
        return loss_fun


class AttentionClassifierPytorch(GenericClassifier):
    def __init__(self,
                 train_config:TrainConfig,
                 model_config: selfatt_enc.BertConfig,
                 dataprocessor: DataProcessorInterface,
                 save_dir='./save_dir',
                 model_file=None,
                 pretrain_file=None,
                 data_parallel=True

                 ) -> None:
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        self.dataprocessor =  dataprocessor
        self.device = get_device()
        self.model = None

        self.save_dir = save_dir
        self.model_file = model_file
        self.pretrain_file = pretrain_file
        self.data_parallel = data_parallel

    def _build_model(self):
        return AttentionClassifier(self.model_config, self.dataprocessor.num_classes)

    def fit(self,X,y=None):
        self.dataprocessor.fit(X)

        model,loss_fun = self.prepare_model_for_training(self.data_parallel, self.model_file, self.pretrain_file)

        # self.optimizer = optim.optim4GPU(self.train_config, self.model) #TODO(tilo):holyJohn!
        if False:
            p_cond = lambda p_name,p: 'bert_encoder' not in p_name and p.requires_grad
        else:
            p_cond = lambda p_name,p: p.requires_grad
        params = [p for p_name, p in self.model.named_parameters() if p_cond(p_name,p)]
        self.optimizer = torch.optim.Adam(params, lr=self.train_config.lr)

        def train_on_batch(batch):
            self.optimizer.zero_grad()
            loss = loss_fun(model,batch).mean()  # mean() for Data Parallelism
            loss.backward()
            self.optimizer.step()
            return loss.item()

        pytorch_methods.train(train_on_batch,
                              self.build_dataloader(X,mode='train',
                                                    batch_size=self.train_config.batch_size),
                              self.train_config.n_epochs,
                              tol=self.train_config.patience,
                              patience=self.train_config.patience,
                              verbose=True)
        return self

    def build_dataloader(self,raw_data,mode,batch_size):
        dataloader = build_messaging_DataLoader_from_dataset_builder(
            dataset_builder=lambda _: GetBatchFunDatasetWrapper(
                self.dataprocessor.build_get_batch_fun(raw_data, batch_size=batch_size)),
            message_supplier=lambda: mode,
            collate_fn=lambda x: to_torch_to_cuda(x[0]),
            num_workers=0
        )
        return dataloader

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
        self.model.eval()
        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)

        with torch.no_grad():
            tmp = pytorch_methods.predict_with_softmax(
                pytorch_nn_model=model,
                batch_iterable=self.build_dataloader(X, batch_size=1024, mode='eval'))
            probas = np.array(list(tmp)).astype('float64')
        return probas

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



