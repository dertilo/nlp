import os
import sys
sys.path.append('.')
from scipy.sparse import csr_matrix #TODO(tilo): if not imported before torch it throws: ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found

from pytorch_util.pytorch_DataLoaders import GetBatchFunDatasetWrapper
from text_classification.attention_classifier_MRPC import TwoSentDataProcessor


from model_evaluation.classification_metrics import calc_classification_metrics
import time
from collections import Counter
from pprint import pprint

import torch
from commons import data_io, util_methods
from torch import nn as nn

from pytorch_util.multiprocessing_proved_dataloading import build_messaging_DataLoader_from_dataset_builder
from pytorch_util.pytorch_methods import get_device, to_torch_to_cuda
from text_classification.classifiers.common import DataProcessorInterface
import pytorchic_bert.selfattention_encoder as selfatt_enc


class BertEmbedder(object):
    def __init__(self,
                 model_config: selfatt_enc.BertConfig,
                 dataprocessor: DataProcessorInterface,
                 model_file:str,
                 data_parallel=True
                 ) -> None:

        super().__init__()
        self.model_config = model_config
        self.dataprocessor = dataprocessor
        self.device = get_device()
        self.model = None

        self.data_parallel = data_parallel
        self.bert_encoder = selfatt_enc.EncoderStack(self.model_config)

        loaded = torch.load(model_file, map_location=None if torch.cuda.is_available() else 'cpu')
        self.bert_encoder.load_state_dict(loaded)

    def build_dataloader(self, raw_data, mode, batch_size):
        dataloader = build_messaging_DataLoader_from_dataset_builder(
            dataset_builder=lambda _: GetBatchFunDatasetWrapper(
                self.dataprocessor.build_get_batch_fun(raw_data, batch_size=batch_size)),
            message_supplier=lambda: mode,
            collate_fn=lambda x: (x[0].pop('raw_batch'), self.prepare_tensors(x)),
            num_workers=0
        )
        return dataloader

    def prepare_tensors(self,x):
        assert 'raw_batch' not in x[0]
        return to_torch_to_cuda(x[0])

    def transform(self, X, batch_size = 32, data_parallel=True):
        self.dataprocessor.fit(X) #TODO(tilo):must be loaded
        self.bert_encoder.eval()
        bert_encoder = self.bert_encoder.to(self.device)
        if data_parallel:
            bert_encoder = nn.DataParallel(bert_encoder)

        dl = self.build_dataloader(X,mode='eval',batch_size=batch_size)
        transformed_data_g = ((raw_batch,bert_encoder(**batch).cpu()) for raw_batch,batch in dl)
        for raw_batch,batch in dl:
            with torch.no_grad():
                tensor = bert_encoder(**batch)
            yield raw_batch, tensor.cpu()
        return transformed_data_g

    def transform_dump(self,X,path,batch_size=32,dump_batch_size=1024):

        if not os.path.isdir(path):
            os.mkdir(path)

        def dump_it(k,batches):
            tensor = torch.cat([tensor for _,tensor in batches], dim=0)
            data_io.write_jsons_to_file(path+'/raw_batch_%d.jsonl'%k,[raw for raw_batch,_ in batches for raw in raw_batch])
            torch.save(tensor, path + '/processed_batch_%d.pt'%k)
        transformed_g = self.transform(X, batch_size)
        for k, batches in enumerate(util_methods.iterable_to_batches(transformed_g, dump_batch_size // batch_size)):
            dump_it(k,batches)


if __name__ == '__main__':
    import numpy as np

    def get_data(file):
        def parse_line(line):
            label, id1, id2, texta, textb = line.split('\t')
            return {
                'text': texta,
                'textb': textb,
                'labels': label}

        lines_g = data_io.read_lines(file)
        next(lines_g)
        data = [parse_line(line) for line in lines_g]
        return data

    def load_data():
        train_data = get_data(home + '/data/glue/MRPC/train.tsv')
        label_counter = Counter([l for d in train_data for l in d['labels']])
        pprint(label_counter)
        test_data = get_data(home + '/data/glue/MRPC/dev.tsv')
        label_counter = Counter([l for d in test_data for l in d['labels']])
        print(label_counter)
        return train_data, test_data

    start = time.time()
    from pathlib import Path
    home = str(Path.home())
    train_data, test_data = load_data()

    model_cfg = selfatt_enc.BertConfig.from_json('pytorchic_bert/config/bert_base.json')
    max_len = 128

    # model_cfg = selfatt_enc.BertConfig(
    #     n_heads=4,
    #     vocab_size=30522,#TODO(tilo):WTF!
    #     dim=32,dim_ff=4*32,n_layers=2)

    vocab = home+'/data/models/uncased_L-12_H-768_A-12/vocab.txt'
    dp = TwoSentDataProcessor(vocab_file=vocab, max_len=max_len)

    pretrain_file = home + '/data/models/uncased_L-12_H-768_A-12/bert_encoder_pytorch.pt'
    # pretrain_file = None
    bertembedder = BertEmbedder(model_cfg,dp,pretrain_file)
    bertembedder.transform_dump(train_data,path='./processed',batch_size=32,dump_batch_size=4096)
    # pbatch = next(bertembedder.transform(train_data))