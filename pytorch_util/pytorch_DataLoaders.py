
from collections import Counter, Iterable
from typing import List, Callable, Dict, Tuple, Any

import numpy as np
import sys
import torch
from commons.util_methods import iterable_to_batches
from torch.autograd import Variable
from torch.utils.data.sampler import SequentialSampler, BatchSampler, WeightedRandomSampler, Sampler


class PyTorchFromListDataset(torch.utils.data.Dataset):
    def __init__(self, data_target_tuples:List,
                 ):
        self.data_target_tuples = data_target_tuples

    def __getitem__(self, index=None):
        return self.data_target_tuples[index]
    def __len__(self):
        return len(self.data_target_tuples)


def build_batching_DataLoader_from_iterator_supplier(
        data_supplier: Callable[[], Iterable],
        process_batch_fun:Callable[[List],Any],
        collate_fn,
        num_workers=0, batch_size=32):
    dataset = PyTorchBatchingDatasetFromIteratorSupplier(data_supplier, batch_size=batch_size,
                                                         process_batch_fun=process_batch_fun,
                                                         num_batches_per_epoch=sys.maxsize #TODO: this is somewhat fuckedup!
                                                         )

    # def init_fn(worker_id):
    #     torch.manual_seed(12345)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              pin_memory=False,
                                              # worker_init_fn=init_fn,
                                              num_workers=num_workers,
                                              batch_size=1,
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                              sampler=SequentialSampler(dataset)) #needs to be SequentialSampler with batch_size 1 !!!
    return data_loader


def build_DataLoader(dataset: torch.utils.data.Dataset,collate_fn=lambda x:x[0], num_workers=0,pin_memory=False):
    data_loader = torch.utils.data.DataLoader(dataset,
                                              num_workers=num_workers,
                                              batch_size=1,
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                              pin_memory=pin_memory,
                                              sampler=SequentialSampler(dataset))
    return data_loader

class MessagingSampler(Sampler):

    def __init__(self, message_supplier, num_batches_per_epoch):
        super().__init__(None)
        self.message_supplier = message_supplier
        self.num_batches_per_epoch = num_batches_per_epoch

    def __iter__(self):
        for k in range(self.num_batches_per_epoch):
            yield [self.message_supplier()]
        yield []
        while True:
            yield [self.message_supplier()]

    def __len__(self):
        return self.num_batches_per_epoch

def collate_fn(x):
    if len(x)==1:
        return x[0]
    else:
        raise StopIteration

def build_messaging_DataLoader(dataset: torch.utils.data.Dataset,
                               message_supplier:Callable[[], Any],
                               collate_fn=collate_fn,
                               num_workers=0,
                               pin_memory=False):
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        batch_sampler=MessagingSampler(message_supplier=message_supplier,
                                       num_batches_per_epoch=len(dataset)))
    return data_loader


def calc_balancing_weights_num_samples(data_targets):
    label_counter = Counter((label for d in data_targets for label in d['target']))
    n = sum([val for val in label_counter.values()])
    weights = np.array([1/(label_counter[datum['target'][0]]*n) for datum in data_targets])
    # weights = weights / sum(weights)
    return weights,num_samples

# def collate_fn(batch):
#     processed_batch = process_batch_fun(batch)
#     input_variables = [Variable(torch.from_numpy(b), requires_grad=False) for b in processed_batch['data']]
#
#     if 'target' in processed_batch:
#         target = Variable(torch.from_numpy(processed_batch['target']), requires_grad=False)
#         return input_variables, target
#     else:
#         return input_variables

def build_sampling_DataLoader(data_targets: List,
                              weights,
                              num_samples,
                              num_workers=0,
                              process_batch_fun=None,
                              batch_size=32):

    assert isinstance(data_targets, list)
    dataset = PyTorchFromListDataset(data_targets)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=process_batch_fun,
        batch_sampler=BatchSampler(WeightedRandomSampler(
            weights=weights,
            # replacement=False,#TODO: cannot sample more than number of positives although there are also lots of negatives
            num_samples=num_samples),
            batch_size=batch_size, drop_last=False
        )
    )
    return data_loader

class BatchSupplingDataset(torch.utils.data.Dataset):
    def __init__(self,
                 batch_supplier:Callable,
                 num_batches_per_epoch,
                 ):
        self.batch_supplier=batch_supplier
        self.num_batches_per_epoch = num_batches_per_epoch

    def __getitem__(self, index=None):
        return self.batch_supplier()

    def __len__(self):
        return self.num_batches_per_epoch

class PyTorchBatchingDatasetFromIteratorSupplier(torch.utils.data.Dataset):
    def __init__(self,
                 data_supplier:Callable[[], Iterable],
                 num_batches_per_epoch,
                 batch_size,
                 process_batch_fun:Callable[[List],Any],
                 ):
        self.data_supplier = data_supplier
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.process_batch_fun = process_batch_fun
        self.raw_batch_iterator = iterable_to_batches(self.data_supplier(), self.batch_size)

    def reset_raw_batch_iterator(self):
        self.raw_batch_iterator = iterable_to_batches(self.data_supplier(), self.batch_size)

    def __getitem__(self, index=None):
        if index == 0:
            self.reset_raw_batch_iterator()
        return self.process_batch_fun(next(self.raw_batch_iterator))

    def __len__(self):
        return self.num_batches_per_epoch

if __name__ == '__main__':
    dummpy_data = [{'data': np.zeros((1,9)),'target': [k]} for ks in [[0]*75,[1]*20,[2]*5]for k in ks]
    weights,num_samples = calc_balancing_weights_num_samples(dummpy_data)
    def dummpy_batch_process_fun(data):
        return {'data':np.array([d['data'] for d in data]),
                'target':np.array([d['target'] for d in data])}
    dl = build_sampling_DataLoader(dummpy_data,weights,num_samples,batch_size=100,process_batch_fun=dummpy_batch_process_fun)

    labels_original = [int(label) for d in dummpy_data for label in d['target']]
    labels_balanced = [int(label) for batch in dl for label in batch['target']]
    print(Counter(labels_original))
    print(Counter(labels_balanced))
