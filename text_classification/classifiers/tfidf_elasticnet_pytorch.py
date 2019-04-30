from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from torch.autograd import Variable
from torch.utils.data import DataLoader

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




class TfIdfElasticNetPytorchClf(GenericClassifier):
    def __init__(self,preprocess_fun,alpha = 0.00001,l1_ratio=0.15) -> None:
        super().__init__()
        self.preprocess_fun = preprocess_fun
        self.vectorizer = TfidfVectorizer(sublinear_tf=True,
                                     preprocessor=identity_dummy_method,
                                     tokenizer=identity_dummy_method,
                                     ngram_range=(1, 1),
                                     max_df=0.75, min_df=2,
                                     max_features=30000,
                                     stop_words=None  # 'english'
                                     )
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.target_binarizer = None

    def batch_generator(self, raw_data:List[Dict], batch_size=32,train_mode=False) -> DataLoader:
        def process_inputs(batch_data):
            bow = self.preprocess_fun(batch_data)
            csr = self.vectorizer.transform(bow)
            # assert all([isinstance(x,csr_matrix) for x in batch_data])
            # csr = vstack(batch_data, format='csr')
            return csr.toarray().astype('float32')

        def _process_batch_fun(batch):
            procesed_batch={'input':process_inputs([d['text'] for d in batch])}
            if train_mode:
                procesed_batch['target'] = self.target_binarizer.transform([d['labels'] for d in batch]).astype('float32')
            return procesed_batch

        return pytorch_DataLoaders.build_batching_DataLoader_from_iterator_supplier(
            data_supplier=lambda : raw_data,
            num_workers=0,
            process_batch_fun=_process_batch_fun,
            collate_fn=lambda x:to_torch_to_cuda(x[0]),
            batch_size=batch_size
        )

    def fit(self,X,y=None):
        self.vectorizer.fit(self.preprocess_fun([d['text'] for d in X]))

        self.target_binarizer = MultiLabelBinarizer()
        self.target_binarizer.fit([d['labels'] for d in X])

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

        losses, batch_losses = pytorch_methods.train(
            train_on_batch_fun=train_on_batch,
            batch_generator_supplier=lambda: iter(self.batch_generator(X,batch_size=32,train_mode=True)),
            num_epochs=6,patience=5,tol=-1,
            verbose=1,
        )

        return self

    def predict_proba(self, sequences)->np.ndarray:
        tmp = pytorch_methods.predict_with_softmax(
            pytorch_nn_model=self.model,
            batch_iterable=self.batch_generator(sequences,batch_size=1024))
        return np.array(list(tmp)).astype('float64')

    def predict_proba_encode_targets(self, data):
        probas = self.predict_proba(data)
        targets = self.target_binarizer.transform([d['labels'] for d in data]).astype('int64')
        return probas, targets
