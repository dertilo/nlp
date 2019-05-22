from typing import NamedTuple

import numpy as np
import torch
from commons.util_methods import iterable_to_batches
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn as nn

from pytorch_util import pytorch_methods
from pytorch_util.multiprocessing_proved_dataloading import build_messaging_DataLoader_from_dataset_builder
from pytorch_util.pytorch_DataLoaders import GetBatchFunDatasetWrapper
from pytorch_util.pytorch_methods import get_device, to_torch_to_cuda
from pytorchic_bert.selfattention_encoder import EncoderLayer
from text_classification.classifiers.common import GenericClassifier, DataProcessorInterface


class TrainConfig(NamedTuple):

    seed: int = 42 # random seed
    batch_size: int = 32
    lr: float = 2e-5 # learning rate
    n_epochs: int = 10 # the number of epoch
    tol:float = 0
    patience:int = 3


class Classifier(nn.Module):

    def __init__(self, embedding_size, n_labels):
        super().__init__()
        # self.encoder = EncoderLayer(dim=embedding_size,dim_ff=4*embedding_size,n_heads=2)
        self.conv = nn.Conv2d(in_channels=embedding_size,
                              out_channels=32,
                              kernel_size=(1, 1),
                              padding=0)
        self.pooling = nn.AdaptiveAvgPool2d((10,1))
        self.classifier = nn.Linear(320, n_labels)

    def forward(self,embedding):
        x = self.conv(embedding.permute(0,3,1,2))
        x = self.pooling(x).squeeze()
        # flattened_seqs = embedding.view(embedding.size(0)*embedding.size(1),embedding.size(2),embedding.size(3))
        # encoded = self.encoder(flattened_seqs)
        logits = self.classifier(x.view(x.size(0),32*10))
        return logits

    @staticmethod
    def build_loss_fun(criterion):
        def loss_fun(model,batch):
            target = batch.pop('target')
            logits = model.forward(**batch)
            return criterion(logits, target)
        return loss_fun

class EmbeddedDataProcessor(DataProcessorInterface):

    def __init__(self):
        super().__init__()
        self.target_binarizer = MultiLabelBinarizer()

    def fit(self, data):
        self.target_binarizer.fit([d['labels'] for d in data])
        self.classes = self.target_binarizer.classes_.tolist()
        self.embedding_size = data[0]['embedding'].shape[1]
        self.max_seqlen = data[0]['embedding'].shape[0]

    def build_get_batch_fun(self,raw_data,batch_size):
        utterances_tensor = torch.cat([d.get('embedding').unsqueeze(0) for d in raw_data],dim=0)

        def get_seqs_in_window(idx, before=-8, after=2):
            ids = [idx + i
                    for i in range(before, after)
                    if idx + i < len(raw_data) and idx + i >= 0 and raw_data[idx + i]['debatefile'] == raw_data[idx]['debatefile']]

            padding = torch.zeros((after-before-len(ids),self.max_seqlen,self.embedding_size))
            seqs = utterances_tensor.index_select(0,torch.LongTensor(ids))
            # seqs = torch.cat([raw_data[i]['embedding'].unsqueeze(0) for i in ids])
            return torch.cat((padding,seqs))

        dialogs = torch.cat([get_seqs_in_window(idx, -8, 2).unsqueeze(0) for idx in range(len(raw_data))],dim=0) # this needs lot of memory!
        del utterances_tensor

        def build_batch_generator(batch_size):
            return iterable_to_batches(iter(range(len(raw_data))), batch_size)

        batch_g =[0]
        batch_g[0] = build_batch_generator(batch_size)

        def get_batch(message):
            try:
                batch_idx = next(batch_g[0])
            except StopIteration:
                batch_g[0] = build_batch_generator(batch_size)
                raise StopIteration

            out = {
                # 'embedding':torch.cat([get_seqs_in_window(idx,-8,2).unsqueeze(0) for idx in batch_idx]),
                'embedding':dialogs.index_select(0,torch.LongTensor(batch_idx)),
            }

            if message == 'eval':
                pass
            elif message == 'train':
                out['target'] = torch.tensor([self.classes.index(raw_data[idx]['labels'][0]) for idx in batch_idx], dtype=torch.long)
            else:
                assert False
            return out

        return get_batch

class EmbeddingClassifier(GenericClassifier):
    '''
    classifies sequences of embeddings
    '''
    def __init__(self,
                 train_config:TrainConfig,
                 data_parallel=True
                 ) -> None:
        super().__init__()
        self.train_config = train_config
        self.device = get_device()
        self.model = None
        self.dataprocessor = EmbeddedDataProcessor()
        self.data_parallel = data_parallel

    def _build_model(self):
        return Classifier(self.dataprocessor.embedding_size, self.dataprocessor.num_classes)

    def fit(self,X,y=None):
        self.dataprocessor.fit(X)

        model,loss_fun = self.prepare_model_for_training(self.data_parallel)

        p_cond = lambda p_name,p: p.requires_grad
        names_params = [(p_name,p) for p_name, p in self.model.named_parameters() if p_cond(p_name,p)]
        params = [p for _, p in names_params]
        self.optimizer = torch.optim.RMSprop(params, lr=self.train_config.lr)

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
                              tol=self.train_config.tol,
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

    def prepare_model_for_training(self, data_parallel):
        if self.model is None:
            self.model = self._build_model()

        print(self.model)

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