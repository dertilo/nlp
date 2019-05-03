import multiprocessing
from functools import partial
from typing import Callable, Any

import numpy as np
import torch
from commons.util_methods import iterable_to_batches
from torch.utils.data import Dataset

from pytorch_util.pytorch_DataLoaders import MessagingSampler

global_dataset: Dataset = None

class DatasetWrapper(Dataset):

    def __getitem__(self, index):
        global global_dataset
        return global_dataset[index]

    def __len__(self):
        assert False


def dataset_initializer(worker_id, dataset_builder):
    #     torch.manual_seed(12345)
    global global_dataset
    global_dataset = dataset_builder(worker_id)


def build_messaging_DataLoader_from_dataset_builder(
        dataset_builder: Callable[[int], Dataset],
        message_supplier: Callable[[], Any],
        collate_fn,
        num_workers=0,
):
    sampler = MessagingSampler(message_supplier=message_supplier)
    if num_workers>0:
        data_loader = torch.utils.data.DataLoader(
            DatasetWrapper(),
            pin_memory=False,#???
            worker_init_fn=partial(dataset_initializer, dataset_builder=dataset_builder),
            num_workers=num_workers,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            batch_sampler=sampler
        )
    else: # Dataset in same process
        dataset_initializer(0,dataset_builder)
        data_loader = torch.utils.data.DataLoader(
            DatasetWrapper(),
            pin_memory=False,  # ???
            num_workers=0,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            batch_sampler=sampler
        )

    return data_loader


if __name__ == '__main__':
    class SomeDataset(Dataset):

        def __init__(self,batch_size) -> None:
            super().__init__()
            data = [{'data': np.zeros((1, 3)), 'target': [k]} for ks in [[0] * 75, [1] * 20, [2] * 5] for k in ks]
            self.batch_g = iterable_to_batches(data, batch_size)


        def __getitem__(self, index):
            message = index
            print(multiprocessing.current_process())
            print('received message: ' + str(message))
            return next(self.batch_g)

            pass

        def __len__(self):
            assert False


    counter = 0

    def train_on_batch_fun(batch):
        global counter
        counter += 1
        return 0.111

    dataloader = build_messaging_DataLoader_from_dataset_builder(
        dataset_builder=lambda i: SomeDataset(
            batch_size=32,
        ),
        collate_fn=lambda x: x[0],
        num_workers=1,
        message_supplier=lambda: counter
    )

    for k in [1]:
        data = [train_on_batch_fun(d) for d in dataloader]
        print(data)
        print(len(data))
