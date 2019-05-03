import torch
import torch.nn as nn
import torch.nn.functional as F
from commons.util_methods import iterable_to_batches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from torch.autograd import Variable

from pytorch_util import pytorch_methods, pytorch_DataLoaders
from pytorch_util.pytorch_DataLoaders import DatasetWrapper
from pytorch_util.pytorch_methods import to_torch_to_cuda
from text_classification.classifiers.common import GenericClassifier

import numpy as np

from text_classification.classifiers.tfidf_dataprocessor import TfIdfTextClfDataProcessor

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
    def __init__(self,
                 text_to_bow_fun,
                 alpha = 0.00001, l1_ratio=0.15) -> None:
        super().__init__()
        self.text_to_bow_fun = text_to_bow_fun

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.dataprocessor = TfIdfTextClfDataProcessor(text_to_bow_fun)

    def fit(self, data, y=None):
        self.dataprocessor.fit(data)
        self.model = LinearClfL1L2Regularized(input_dim=len(self.dataprocessor.vectorizer.get_feature_names()),
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
            num_epochs=2,patience=5,tol=-1,
            verbose=1,
        )

        return self

    def build_dataloader(self,data,batch_size,mode):

        self.batch_g = iterable_to_batches(data, batch_size)

        def process_batch_fun(message):

            try:
                batch = next(self.batch_g)
            except StopIteration:
                self.batch_g = iterable_to_batches(data, batch_size)
                raise StopIteration

            if message == 'train':
                inputs, targets = self.dataprocessor.process_inputs_and_targets(batch)
                processed_batch = {'input': inputs.toarray().astype('float32'),
                                   'target': targets}
            elif message == 'eval':
                processed_batch = {'input': self.dataprocessor.process_inputs(batch).toarray().astype('float32')}
            else:
                assert False

            return processed_batch


        dataloader = pytorch_DataLoaders.build_messaging_DataLoader_from_dataset(
            dataset=DatasetWrapper(getbatch_fun=process_batch_fun),
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

    @property
    def target_binarizer(self):
        return self.dataprocessor.target_binarizer
