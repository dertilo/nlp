import abc
from collections import OrderedDict
from typing import Dict, Iterable, List, Callable, Any

import time

import sys

import numpy
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
from torch.utils.data.dataloader import _DataLoaderIter

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print('using: ' + torch.cuda.get_device_name(0))
else:
    print('no GPU found!')

import numpy as np


# def train_on_batch(batch):
#     loss = self.model.loss(**batch)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     return loss.data

def iterate_and_time(g):
    d = next(g)
    while True:
        start = time.time()
        yield d, time.time() - start
        try:
            d = next(g)
        except StopIteration:
            break

def train(
        train_on_batch_fun,
        batch_generator_supplier:Callable[[],Any],
        num_epochs = 10,
        patience = 2,
        tol = 0.01,
        verbose=False,
):
    losses = []; all_batch_losses = []
    start = time.time()
    for t in range(num_epochs):
        iterator = batch_generator_supplier()
        batch_start = time.time()
        batch_losses_durations = [(train_on_batch_fun(batch), dur) for batch, dur in iterate_and_time(iterator)]
        batch_losses = [bl for bl,_ in batch_losses_durations]
        all_batch_losses.append(batch_losses)
        wait_time = np.sum([d for _, d in batch_losses_durations])
        assert len(batch_losses)>0
        mean_loss = np.mean(batch_losses)
        if verbose:
            print('epoch: %d; num batches: %d; mean-loss: %0.4f; duration: %0.2f seconds; waiting for data-loader: %0.4f'
                  %(t,len(batch_losses),float(mean_loss),float(time.time()-batch_start),float(wait_time)))
            sys.stdout.flush()
        losses.append(mean_loss)
        if t>=patience and 1-losses[-1]/losses[t-patience]<tol:
            if verbose: print('stopped early due to non-decreasing loss')
            break
    if verbose:
        print('training took: %d seconds'%(time.time()-start))
        sys.stdout.flush()
    return losses,all_batch_losses


def to_torch_to_cuda(x):
    if isinstance(x,dict):
        return {name: to_cuda(torch.from_numpy(b)) if isinstance(b,numpy.ndarray) else to_cuda(b)
                for name, b in x.items() }
    else:
        return to_cuda(torch.from_numpy(x))

def to_cuda(x):
    if isinstance(x,dict):
        if USE_CUDA:
            return {n:x.cuda() for n,x in x.items() if isinstance(x,torch.Tensor) or isinstance(x,Variable)}
        else:
            return x
    else:
        return x.cuda() if USE_CUDA else x

def predict_with_softmax(pytorch_nn_model:nn.Module, batch_iterable:Iterable[Dict[str,Variable]]):
    pytorch_nn_model.eval()
    return (p for batch in batch_iterable
                for p in F.softmax(pytorch_nn_model(**to_cuda(batch)), dim=1).data.cpu().numpy().tolist())

def predict_rawscores(pytorch_nn_model:nn.Module, batch_iterable:Iterable[Dict[str,Variable]]):
    pytorch_nn_model.eval()
    return (p for variables in batch_iterable
                for p in pytorch_nn_model(**to_cuda(variables)).data.cpu().numpy().tolist())

def predict_sigmoid(pytorch_nn_model:nn.Module, batch_iterable:Iterable[Dict[str,Variable]]):
    pytorch_nn_model.eval()
    return (p for variables in batch_iterable
                for p in F.sigmoid(pytorch_nn_model(**to_cuda(variables))).data.cpu().numpy().tolist())


def data_dict_to_torch_variables(data:Dict):
    return {name: Variable(torch.from_numpy(b), requires_grad=False) for name, b in
                  data.items() if isinstance(b,numpy.ndarray)}

def data_dict_to_torch(data:Dict):
    return {name: torch.from_numpy(b) for name, b in
                  data.items() if isinstance(b,numpy.ndarray)}

def print_module(model:torch.nn.Module):
    print(model)
    print('parameter shapes: ')
    for name, param in model.named_parameters():
        is_cuda = '-> CUDA' if param.is_cuda else '-> cpu'
        trainable = 'trainable' if param.requires_grad else ''
        print(name + ': ' + str(param.data.shape) + ' ' + is_cuda + ' ' + trainable)
    print('* number of parameters: %d' % sum([p.nelement() for p in model.parameters()]))


