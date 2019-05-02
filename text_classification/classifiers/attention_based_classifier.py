import json
import os
from collections import Counter
from pprint import pprint
from typing import NamedTuple

import torch
from commons.util_methods import iterable_to_batches
from torch import nn as nn
from torch.utils.data import Dataset

from getting_data.clef2019 import get_Clef2019_data
from pytorch_util.multiprocessing_proved_dataloading import build_messaging_DataLoader_from_dataset_builder
from pytorch_util.pytorch_methods import get_device
from pytorchic_bert import checkpoint
from pytorchic_bert import tokenization
from pytorchic_bert.preprocessing import Pipeline, SentencePairTokenizer, AddSpecialTokensWithTruncation, TokenIndexing
from pytorchic_bert.utils import set_seeds
from text_classification.classifiers.common import GenericClassifier


class Config(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    batch_size: int = 32
    lr: int = 5e-5 # learning rate
    n_epochs: int = 10 # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup: float = 0.1
    save_steps: int = 100 # interval for saving model
    total_steps: int = 100000 # total number of steps to train

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))


class AttentionClassifier(nn.Module):

    def __init__(self, cfg, n_labels):
        super().__init__()
        # self.transformer = selfatt_enc.Baseline(cfg)
        self.transformer = selfatt_enc.EncoderStack(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.classifier = nn.Linear(cfg.dim, n_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        h = self.transformer(input_ids, segment_ids, input_mask)
        # only use the first h in the sequence
        pooled_h = torch.tanh(self.fc(h[:, 0]))
        logits = self.classifier(self.drop(pooled_h))
        return logits


# class BaselineClf(nn.Module):
#     def __init__(self, cfg, n_labels):
#         super().__init__()
#         self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim) # token embedding
#         self.classifier = nn.Linear(cfg.dim, n_labels)
#
#     def forward(self, x, seg, mask):
#         embedded = self.tok_embed(x)
#         return self.classifier(torch.sum(embedded, dim=1))

class PipelineProcessingDataset(Dataset):

    def __init__(self,
                 raw_data,
                 vocab_file,
                 class_labels,
                 max_len,
                 batch_size):

        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
        self.pipeline = Pipeline([SentencePairTokenizer(tokenizer.convert_to_unicode, tokenizer.tokenize),
                             AddSpecialTokensWithTruncation(max_len),
                             TokenIndexing(tokenizer.convert_tokens_to_ids, class_labels, max_len)
                             ]
                            )
        data = [self.pipeline.transform((d['label'],d['text'],'')) for d in raw_data]
        self.tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]
        self.batch_indizes_g = iterable_to_batches(iter(range(self.tensors[0].shape[0])), batch_size)

    def __getitem__(self, index):
        batch = next(self.batch_indizes_g)
        return tuple(tensor[batch] for tensor in self.tensors)

    def __len__(self):
        assert False

class AttentionClassifierPipeline(GenericClassifier):
    def __init__(self,cfg,
                 model_cfg,
                 vocab_file,
                 max_len,
                 class_labels
                 ) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.cfg = cfg
        self.vocab_file = vocab_file
        self.max_len = max_len
        self.class_labels = class_labels
        self.num_labels = len(class_labels)

    def fit(self,X,y=None,
            save_dir='./save_dir',
            model_file = None,
            pretrain_file = None,
            data_parallel = True
            ):
        self.save_dir = save_dir
        data_iter = build_messaging_DataLoader_from_dataset_builder(
            dataset_builder=lambda i:PipelineProcessingDataset(X,self.vocab_file,self.class_labels,self.max_len,batch_size=32),
            message_supplier=lambda :None,
            collate_fn=lambda x:x[0],
            num_workers=1
        )
        self.model = AttentionClassifier(self.model_cfg, self.num_labels)
        print(self.model)
        # optimizer = optim.optim4GPU(self.cfg, model) #TODO(tilo):holyJohn!
        self.optimizer = torch.optim.RMSprop([p for p in self.model.parameters() if p.requires_grad], lr=0.01)

        criterion = nn.CrossEntropyLoss()
        def get_loss(model, batch, global_step): # make sure loss is a scalar tensor
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            loss = criterion(logits, label_id)
            return loss

        self.train(data_iter,get_loss, model_file, pretrain_file, data_parallel)
        return self

    def predict_proba(self,X):

        def pred_batch_fun(model, batch):
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            return logits

        data_iter = build_messaging_DataLoader_from_dataset_builder(
            dataset_builder=lambda i:PipelineProcessingDataset(X,self.vocab_file,self.class_labels,self.max_len,batch_size=32),
            message_supplier=lambda :None,
            collate_fn=lambda x:x[0],
            num_workers=1
        )

        batch_results = self.eval(data_iter,pred_batch_fun,model_file=None)
        results= torch.cat(batch_results)
        return results.cpu().numpy().astype('float64')

    def predict_proba_encode_targets(self, data):
        pass


    def train(self,data_iter, get_loss, model_file=None, pretrain_file=None, data_parallel=True):
        self.device = get_device()
        self.model.train()
        self.load(model_file, pretrain_file)
        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)

        global_step = 0 # global iteration steps regardless of epochs
        for e in range(self.cfg.n_epochs):
            loss_sum = 0. # the sum of iteration losses to get average loss in every epoch
            # iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')
            iter_bar = data_iter
            for i, batch in enumerate(iter_bar):
                batch = [t.to(self.device) for t in batch]

                self.optimizer.zero_grad()
                loss = get_loss(model, batch, global_step).mean() # mean() for Data Parallelism
                loss.backward()
                self.optimizer.step()

                global_step += 1
                loss_sum += loss.item()
                # iter_bar.set_description('Iter (loss=%5.3f)'%loss.item())

                if global_step % self.cfg.save_steps == 0: # save
                    self.save(global_step)

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
                    print('The Total Steps have been reached.')
                    self.save(global_step) # save and finish when global_steps reach total_steps
                    return

            print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
        self.save(global_step)

    def eval(self, data_iter, evaluate, model_file, data_parallel=True):

        self.model.eval() # evaluation mode
        self.load(model_file, None)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        results = [] # prediction results
        # iter_bar = tqdm(data_iter, desc='Iter (loss=X.XXX)')
        iter_bar = data_iter
        with torch.no_grad():  # evaluation without gradient calculation
            for batch in iter_bar:
                batch = [t.to(self.device) for t in batch]
                result = evaluate(model, batch) # accuracy to print
                results.append(result)

        return results

    def load(self, model_file, pretrain_file):
        if model_file:
            print('Loading the model from', model_file)
            self.model.load_state_dict(torch.load(model_file))

        elif pretrain_file: # use pretrained transformer
            print('Loading the pretrained model from', pretrain_file)
            if pretrain_file.endswith('.ckpt'): # checkpoint file in tensorflow
                checkpoint.load_model(self.model.transformer, pretrain_file)
            elif pretrain_file.endswith('.pt'): # pretrain model file in pytorch
                self.model.transformer.load_state_dict(
                    {key[12:]: value
                        for key, value in torch.load(pretrain_file).items()
                        if key.startswith('transformer')}
                ) # load only transformer parts


    def save(self, i):
        torch.save(self.model.state_dict(), # save model object before nn.DataParallel
            os.path.join(self.save_dir, 'model_steps_'+str(i)+'.pt'))

if __name__ == '__main__':

    from pathlib import Path
    home = str(Path.home())
    cwd = home
    data = [{'text':d['utterance'],'label':str(d['label'])} for d in get_Clef2019_data(cwd+'/data/fact_checking/clef2019')]

    label_counter = Counter([d['label'] for d in data])
    pprint(label_counter)

    model_cfg = 'config/bert_tiny.json'
    cfg = Config(lr=1e-4,n_epochs=1,batch_size=128)
    import pytorchic_bert.selfattention_encoder as selfatt_enc
    model_cfg = selfatt_enc.Config.from_json(model_cfg)
    max_len = model_cfg.max_len

    set_seeds(cfg.seed)

    vocab = cwd+'/data/models/uncased_L-12_H-768_A-12/vocab.txt'

    pipeline = AttentionClassifierPipeline(cfg,model_cfg,vocab,max_len,list(label_counter.keys()))
    pipeline.fit(data)
    proba = pipeline.predict_proba(data)
