import sys

from text_classification.MRPC_dataprocessing import TwoSentDataProcessor

sys.path.append('.')

from sklearn.model_selection import ShuffleSplit

from getting_data.sentiment140 import get_sentiment140_data


from text_classification.classifiers.attention_based_classifier import TrainConfig, AttentionClassifierPytorch

from model_evaluation.classification_metrics import calc_classification_metrics
import time
from collections import Counter
from pprint import pprint

from commons import data_io

from pytorchic_bert.utils import set_seeds
import pytorchic_bert.selfattention_encoder as selfatt_enc


if __name__ == '__main__':
    import numpy as np

    def evaluate(pipeline1, train_data, test_data):
        target_names = pipeline1.target_binarizer.classes_.tolist()

        proba, y_train = pipeline1.predict_proba_encode_targets(train_data)
        pred = np.array(proba == np.expand_dims(np.max(proba, axis=1), 1), dtype='int64')
        train_scores = calc_classification_metrics(proba, pred, y_train, target_names=target_names)
        pprint('Train-f1-micro: %0.2f' % train_scores['f1-micro'])

        proba, y_test = pipeline1.predict_proba_encode_targets(test_data)
        pred = np.array(proba == np.expand_dims(np.max(proba, axis=1), 1), dtype='int64')
        test_scores = calc_classification_metrics(proba, pred, y_test, target_names=target_names)
        pprint('Test-f1-micro: %0.2f' % test_scores['f1-micro'])

        print('evaluating took: %0.2f secs' % (time.time() - start))


    def load_data():
        max_text_len=20
        data = [{'text': d['text'][:min(len(d['text']), max_text_len)], 'labels': [str(d['polarity'])],'textb':''} for d in
                get_sentiment140_data(datafile=home + '/data/Sentiment140/training.1600000.processed.noemoticon.csv',
                                      )]
        data = [data[i] for i in np.random.random_integers(0,len(data)-1,size=(10_000,))]

        splitter = ShuffleSplit(n_splits=1, test_size=0.9, random_state=111)
        splits = [(train, test) for train, test in splitter.split(X=range(len(data)))]

        train_data = [data[i] for i in splits[0][0]]
        test_data = [data[i] for i in splits[0][1]]

        label_counter = Counter([l for d in train_data for l in d['labels']])
        pprint(label_counter)
        label_counter = Counter([l for d in test_data for l in d['labels']])
        print(label_counter)
        return train_data, test_data

    start = time.time()
    from pathlib import Path
    home = str(Path.home())
    train_data, test_data = load_data()

    cfg = TrainConfig(seed=42,n_epochs=1)#.from_json('pytorchic_bert/config/train_mrpc.json')
    model_cfg = selfatt_enc.BertConfig(
        n_heads=8,
        vocab_size=30522,#TODO(tilo):WTF!
        dim=32,dim_ff=4*32,n_layers=2)
    max_len = 128

    set_seeds(cfg.seed)

    vocab = home+'/data/models/uncased_L-12_H-768_A-12/vocab.txt'
    dp = TwoSentDataProcessor(vocab_file=vocab, max_len=max_len)
    pipeline = AttentionClassifierPytorch(cfg, model_cfg,dp)
    # pretrain_file = home + '/data/models/uncased_L-12_H-768_A-12/bert_encoder_pytorch.pt'
    pretrain_file = None
    save_path = home + '/code/NLP/nlp/saved_models'
    pipeline.fit(train_data, pretrain_file=pretrain_file)
    pipeline.save(save_path)
    del pipeline

    pipeline = AttentionClassifierPytorch.load(save_path)
    evaluate(pipeline,train_data,test_data)
