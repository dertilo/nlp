from collections import Counter
from typing import List, Callable, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SequentialSampler, BatchSampler, WeightedRandomSampler, Sampler


class PyTorchFromListDataset(torch.utils.data.Dataset):
    def __init__(self, data_target_tuples:List):
        self.data_target_tuples = data_target_tuples

    def __getitem__(self, index=None):
        return self.data_target_tuples[index]
    def __len__(self):
        return len(self.data_target_tuples)

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


class MessagingSampler(Sampler):

    def __init__(self, message_supplier):
        super().__init__(None)
        self.message_supplier = message_supplier

    def __iter__(self):
        while True:
            yield [self.message_supplier()]

    def __len__(self):
        assert False

def build_messaging_DataLoader_from_dataset(
        dataset:Dataset,
        collate_fn,
        num_workers=0,# to be used with caution!
        message_supplier: Callable[[], Any]=lambda:None,

):
    if num_workers>0:
        raise Warning('caution! this is pickling the dataset-object!')

    data_loader = torch.utils.data.DataLoader(
        dataset,
        pin_memory=False,#???
        num_workers=num_workers,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        batch_sampler=MessagingSampler(message_supplier=message_supplier)
    )
    return data_loader

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
