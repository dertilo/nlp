from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from commons.util_methods import iterable_to_batches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from torch.autograd import Variable
from torch.utils.data import Dataset

from pytorch_util import pytorch_methods, pytorch_DataLoaders
from pytorch_util.pytorch_methods import to_torch_to_cuda
from text_classification.classifiers.common import GenericClassifier

import numpy as np

from text_classification.classifiers.tfidf_sgd_sklearn import identity_dummy_method

USE_CUDA = torch.cuda.is_available()

class LinearClfL1L2Regularized(nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes,
                 alpha=0.001,
                 l1_ratio=0.15,
                 is_multilabel=False,
                 ):
        super().__init__()
        self.linear_layer = nn.Linear(input_dim, num_classes)
        self.parameters_w = next(self.linear_layer.parameters())
        self.objective = nn.BCEWithLogitsLoss() if is_multilabel else nn.CrossEntropyLoss()
        self.l1_penalty = l1_ratio * alpha
        self.l2_penalty = (1 - l1_ratio) * alpha

    def loss(self,input:Variable,target:Variable):
        target = torch.argmax(target, dim=1) if isinstance(self.objective,nn.CrossEntropyLoss) else target
        nn_output = self.forward(input)
        w = torch.zeros_like(self.parameters_w)
        l1_loss = F.l1_loss(self.parameters_w, w).cpu()
        l2_loss = F.mse_loss(self.parameters_w, w).cpu()
        classification_loss = self.objective.forward(nn_output, target).cpu()
        l1l2_regul_loss = self.l1_penalty * l1_loss + self.l2_penalty * l2_loss
        return classification_loss + l1l2_regul_loss

    def forward(self, input:Variable):
        return self.linear_layer(input)


class BatchingDataset(Dataset):

    def __init__(self,
                 data_supplier,
                 text_to_bow_fun,
                 vectorizer,
                 target_binarizer,
                 batch_size=32) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.data_supplier = data_supplier
        self.text_to_bow_fun=text_to_bow_fun
        self.target_binarizer = target_binarizer
        self.vectorizer = vectorizer
        self.batch_g = iterable_to_batches(data_supplier(), batch_size)

    def __getitem__(self, index):
        message = index
        if message == 'train':
            is_train = True
        elif message == 'eval':
            is_train = False
        else:
            assert False
        try:
            batch = next(self.batch_g)
        except StopIteration:
            self.batch_g = iterable_to_batches(self.data_supplier(), self.batch_size)
            raise StopIteration

        return self._process_batch_fun(batch, is_train=is_train)

    def process_inputs(self,batch_data):
        bow = self.text_to_bow_fun(batch_data)
        csr = self.vectorizer.transform(bow)
        # assert all([isinstance(x,csr_matrix) for x in batch_data])
        # csr = vstack(batch_data, format='csr')
        return csr.toarray().astype('float32')

    def _process_batch_fun(self,batch,is_train):
        procesed_batch = {'input': self.process_inputs([d['text'] for d in batch])}
        if is_train:
            procesed_batch['target'] = self.target_binarizer.transform([d['labels'] for d in batch]).astype('float32')
        return procesed_batch

    def __len__(self):
        assert False


class TfIdfElasticNetPytorchClf(GenericClassifier):
    def __init__(self,
                 text_to_bow_fun,
                 alpha = 0.00001, l1_ratio=0.15) -> None:
        super().__init__()
        self.text_to_bow_fun = text_to_bow_fun

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.target_binarizer = None

    def fit(self, data, y=None):
        self.vectorizer = TfidfVectorizer(sublinear_tf=True,
                                     preprocessor=identity_dummy_method,
                                     tokenizer=identity_dummy_method,
                                     ngram_range=(1, 1),
                                     max_df=0.75, min_df=2,
                                     max_features=30000,
                                     stop_words=None  # 'english'
                                     )

        self.vectorizer.fit(self.text_to_bow_fun([d['text'] for d in data]))

        self.target_binarizer = MultiLabelBinarizer()
        self.target_binarizer.fit([d['labels'] for d in data])

        self.model = LinearClfL1L2Regularized(input_dim=len(self.vectorizer.get_feature_names()),
                                              num_classes=len(self.target_binarizer.classes_.tolist()),
                                              alpha=self.alpha, l1_ratio=self.l1_ratio, is_multilabel=False
                                              )
        optimizer = torch.optim.RMSprop([p for p in self.model.parameters() if p.requires_grad], lr=0.01)

        if USE_CUDA:
            self.model.cuda()

        def train_on_batch(batch):
            loss = self.model.loss(**batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss.data

        dataloader = self.build_dataloader(data,batch_size=32,mode='train')

        losses, batch_losses = pytorch_methods.train(
            train_on_batch_fun=train_on_batch,
            dataloader=dataloader,
            num_epochs=6,patience=5,tol=-1,
            verbose=1,
        )

        return self

    def build_dataloader(self,data,batch_size,mode):
        dataset = BatchingDataset(data_supplier=lambda: data,
                                  text_to_bow_fun=self.text_to_bow_fun,
                                  vectorizer=self.vectorizer,
                                  target_binarizer=self.target_binarizer,
                                  batch_size=batch_size
                                  )
        dataloader = pytorch_DataLoaders.build_messaging_DataLoader_from_dataset(
            dataset=dataset,
            collate_fn=lambda x: to_torch_to_cuda(x[0]),
            num_workers=0,
            message_supplier=lambda: mode
        )
        return dataloader

    def predict_proba(self, data)->np.ndarray:

        dataloader = self.build_dataloader(data,batch_size=128,mode='eval')

        tmp = pytorch_methods.predict_with_softmax(
            pytorch_nn_model=self.model,
            batch_iterable=dataloader)
        return np.array(list(tmp)).astype('float64')

    def predict_proba_encode_targets(self, data):
        probas = self.predict_proba(data)
        targets = self.target_binarizer.transform([d['labels'] for d in data]).astype('int64')
        return probas, targets
