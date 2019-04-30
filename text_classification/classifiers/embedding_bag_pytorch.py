from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MultiLabelBinarizer

from torch.autograd import Variable

import numpy as np
from torch.utils.data import DataLoader

from pytorch_util import pytorch_methods, pytorch_DataLoaders
from pytorch_util.pytorch_methods import to_torch_to_cuda
from text_classification.classifiers.common import GenericClassifier
from text_processing.text_processor import TokenizeIndexer, PADDING

USE_CUDA = torch.cuda.is_available()

def process_inputs(batch_data):
    assert all([len(d['data'])>0 for d in batch_data])
    offsets = np.array(np.cumsum([0]+[len(d['data']) for d in batch_data]).tolist()[:-1])
    batch_concatenated_idx_seqs = np.array([idx for d in batch_data for idx in d['data']])
    assert max(offsets)<len(batch_concatenated_idx_seqs)
    return {
        'concated_batch_idx_seqs':batch_concatenated_idx_seqs.astype('int64'),
        'seq_start_offsets':offsets.astype('int64')}

class BoeClfPytorchModule(nn.Module):
    def __init__(self,
                 num_classes,
                 vocab_size=10,
                 dropout_p = 0.5,
                 embedding_dim=3,
                 alpha=1.0,
                 l1_ratio=0.2,
                 is_multilabel=False
                 ):
        super().__init__()
        self.embedding_mean = nn.EmbeddingBag(vocab_size, embedding_dim, mode='mean')
        self.linear = nn.Linear(embedding_dim, num_classes)
        self.dropout= nn.Dropout(p=dropout_p)


        self.l1_penalty = l1_ratio * alpha
        self.l2_penalty = (1 - l1_ratio) * alpha
        self.objective = nn.BCEWithLogitsLoss(reduce=True) if is_multilabel else nn.CrossEntropyLoss(reduce=True)


    def forward(self, concated_batch_idx_seqs:Variable,seq_start_offsets:Variable,**kwargs_to_ignore):
        text_embedding = self.embedding_mean(concated_batch_idx_seqs, seq_start_offsets)
        return self.linear(self.dropout(text_embedding))

    def loss(self, concated_batch_idx_seqs:Variable,seq_start_offsets:Variable, target: Variable):
        loss_missclass = self._clf_loss(concated_batch_idx_seqs, seq_start_offsets, target).cpu()
        return loss_missclass+self._l1l2regulated_loss()

    def _clf_loss(self, concated_batch_idx_seqs:Variable,seq_start_offsets:Variable, target: Variable):
        target = torch.tensor(target==1.0,dtype=torch.float32)
        target = torch.argmax(target, dim=1) if isinstance(self.objective,nn.CrossEntropyLoss) else target
        if USE_CUDA: target = target.cuda()
        nn_output = self.forward(concated_batch_idx_seqs,seq_start_offsets)
        m_loss_misclass = self.objective(nn_output, target)
        return m_loss_misclass


    def _l1l2regulated_loss(self):
        parameters_w = self.embedding_mean.weight
        w = torch.zeros_like(parameters_w)
        if USE_CUDA: w = w.cuda()
        l1_loss = F.l1_loss(parameters_w, w).cpu()
        l2_loss = F.mse_loss(parameters_w, w).cpu()
        return self.l1_penalty * l1_loss + self.l2_penalty * l2_loss


class EmbeddingBagClassifier(GenericClassifier):
    def __init__(self,embedding_dim=3, alpha = 0.00001,l1_ratio=0.15) -> None:
        super().__init__()
        self.tokenizendexer = TokenizeIndexer()

        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.target_binarizer = None

    def batch_generator(self, raw_data:List[Dict], batch_size=32,train_mode=False) -> DataLoader:
        def impute_PADDING_if_empty(seq):
            if len(seq)==0:
                seq = [self.tokenizendexer.token2key[PADDING]]
            return seq

        def process_inputs(batch_data):
            seqs = [impute_PADDING_if_empty(self.tokenizendexer.transform_to_seq(text)) for text in batch_data]
            offsets = np.array(np.cumsum([0] + [len(seq) for seq in seqs]).tolist()[:-1])
            batch_concatenated_idx_seqs = np.array([idx for seq in seqs for idx in seq])
            assert max(offsets) < batch_concatenated_idx_seqs.shape[0]
            return {
                'concated_batch_idx_seqs': batch_concatenated_idx_seqs.astype('int64'),
                'seq_start_offsets': offsets.astype('int64')}

        def _process_batch_fun(batch):
            procesed_batch=process_inputs([d['text'] for d in batch])
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
        self.tokenizendexer.fit((d['text'] for d in X))

        self.target_binarizer = MultiLabelBinarizer()
        self.target_binarizer.fit([d['labels'] for d in X])

        self.model = BoeClfPytorchModule(
            embedding_dim=self.embedding_dim,
            num_classes=len(self.target_binarizer.classes_.tolist()),
            vocab_size=self.tokenizendexer.get_vocab_size(),
            alpha=self.alpha, l1_ratio=self.l1_ratio, is_multilabel=True
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
            num_epochs=15,patience=5,tol=-1,
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

