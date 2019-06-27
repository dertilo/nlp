import os
import shutil

import numpy
from commons import data_io, util_methods
from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus


if __name__ == '__main__':

    g = (d['pdf_full_text'].replace('\n','') for d in data_io.read_jsons_from_file('../data/arxiv.jsonl.gz')
         if isinstance(d['pdf_full_text'],str) and len(d['pdf_full_text'])>10)

    corpus_path = '../data/corpus'
    if os.path.isdir(corpus_path):
        shutil.rmtree(corpus_path)
    if not os.path.exists(corpus_path): os.mkdir(corpus_path)
    num_train_docs = 9000
    train_file = '%s/train.txt'% corpus_path
    data_io.write_to_file(train_file, (next(g) for k in range(num_train_docs)))
    train_split_folder = corpus_path + '/train'
    if not os.path.exists(train_split_folder): os.mkdir(train_split_folder)
    train_lines_g = data_io.read_lines(train_file)
    docs_per_split = 100
    num_train_splits = int(numpy.ceil(num_train_docs / docs_per_split))
    for k in range(num_train_splits):
        split_file = train_split_folder + '/train_split_%d'%k
        data_io.write_to_file(split_file, (next(g) for k in range(docs_per_split)))

    # shutil.copy(train_file, train_split_folder + '/train_split_%d'%k)
    data_io.write_to_file('%s/valid.txt' % corpus_path, (next(g) for k in range(100)))
    data_io.write_to_file('%s/test.txt' % corpus_path, (next(g) for k in range(100)))

    is_forward_lm = True
    dictionary: Dictionary = Dictionary.load('chars')

    corpus = TextCorpus(corpus_path,
                        dictionary,
                        is_forward_lm,
                        character_level=True)

    language_model = LanguageModel(dictionary,
                                   is_forward_lm,
                                   hidden_size=1024,
                                   nlayers=1)

    trainer = LanguageModelTrainer(language_model, corpus)

    path = 'flair_resources/language_model'
    trainer.train(path,checkpoint=True,
                  sequence_length=10,
                  mini_batch_size=32,
                  max_epochs=10)