from collections import Counter
from pprint import pprint

import torch
from sklearn.base import BaseEstimator
from torch import nn as nn
from torch.utils.data import DataLoader

from getting_data.clef2019 import get_Clef2019_data
from pytorchic_bert import data_loading
from pytorchic_bert import optim
from pytorchic_bert import selfattention_encoder as selfatt_enc
from pytorchic_bert import tokenization
from pytorchic_bert import training
from pytorchic_bert.preprocessing import Pipeline, SentencePairTokenizer, AddSpecialTokensWithTruncation, TokenIndexing
from pytorchic_bert.utils import get_device, set_seeds


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


class BaselineClf(nn.Module):
    def __init__(self, cfg, n_labels):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim) # token embedding
        self.classifier = nn.Linear(cfg.dim, n_labels)

    def forward(self, x, seg, mask):
        embedded = self.tok_embed(x)
        return self.classifier(torch.sum(embedded, dim=1))


class AttentionClassifierPipeline(BaseEstimator):
    def __init__(self,cfg,
                 model_cfg,
                 vocab_file,
                 max_len,
                 class_labels
                 ) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.cfg = cfg

        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
        self.num_labels = len(class_labels)
        self.pipeline = Pipeline([SentencePairTokenizer(tokenizer.convert_to_unicode, tokenizer.tokenize),
                             AddSpecialTokensWithTruncation(max_len),
                             TokenIndexing(tokenizer.convert_tokens_to_ids, class_labels, max_len)
                             ]
                            )


    def fit(self,X,y=None,
            save_dir='./save_dir',
            model_file = None,
            pretrain_file = None,
            data_parallel = True
            ):
        self.trainer = self._build_trainer(X, save_dir)

        criterion = nn.CrossEntropyLoss()
        def get_loss(model, batch, global_step): # make sure loss is a scalar tensor
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            loss = criterion(logits, label_id)
            return loss

        self.trainer.train(get_loss, model_file, pretrain_file, data_parallel)
        return self

    def _build_trainer(self, X, save_dir):
        dataset = data_loading.TextLabelTupleDataset(raw_data=X, pipeline=self.pipeline)
        data_iter = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)
        # model = AttentionClassifier(self.model_cfg, self.num_labels)
        model = BaselineClf(self.model_cfg, self.num_labels)
        print(model)
        # optimizer = optim.optim4GPU(self.cfg, model)
        optimizer = torch.optim.RMSprop([p for p in model.parameters() if p.requires_grad], lr=0.01)


        trainer = training.Trainer(self.cfg,
                                   model,
                                   data_iter,
                                   optimizer,
                                   save_dir, get_device())
        return trainer

    # def evaluate(self,X,
    #              save_dir='./save_dir',
    #              model_file=None,
    #              data_parallel=True
    #              ):
    #
    #     def evaluate(model, batch):
    #         input_ids, segment_ids, input_mask, label_id = batch
    #         logits = model(input_ids, segment_ids, input_mask)
    #         _, label_pred = logits.max(1)
    #         result = (label_pred == label_id).float() #.cpu().numpy()
    #         accuracy = result.mean()
    #         return accuracy, result
    #
    #     results = self.trainer.eval(evaluate, model_file, data_parallel)
    #     total_accuracy = torch.cat(results).mean().item()
    #     print('Accuracy:', total_accuracy)

    def predict_proba(self,X):

        def pred_batch_fun(model, batch):
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            return logits

        dataset = data_loading.TextLabelTupleDataset(raw_data=X, pipeline=self.pipeline)
        batch_results = self.trainer.eval(dataset,pred_batch_fun,model_file=None)
        results= torch.cat(batch_results)
        return results.cpu().numpy().astype('float64')

if __name__ == '__main__':

    from pathlib import Path
    home = str(Path.home())
    cwd = home
    data = [{'text':d['utterance'],'label':str(d['label'])} for d in get_Clef2019_data(cwd+'/data/fact_checking/clef2019')]

    label_counter = Counter([d['label'] for d in data])
    pprint(label_counter)

    model_cfg = 'config/bert_tiny.json'
    cfg = training.Config(lr=1e-4,n_epochs=2,batch_size=32)
    import selfattention_encoder as selfatt_enc
    model_cfg = selfatt_enc.Config.from_json(model_cfg)
    max_len = model_cfg.max_len

    set_seeds(cfg.seed)

    vocab = cwd+'/data/models/uncased_L-12_H-768_A-12/vocab.txt'

    pipeline = AttentionClassifierPipeline(cfg,model_cfg,vocab,max_len,list(label_counter.keys()))
    pipeline.fit(data)
    proba = pipeline.predict_proba(data)
