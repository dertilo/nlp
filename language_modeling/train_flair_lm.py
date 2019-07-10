import sys
sys.path.append('.')
from scipy.sparse import csr_matrix

from getting_data.parse_opensubtitles import build_opensubtitles_lines_generator
from language_modeling.language_model import LanguageModel

import os
import shutil
import torch

import numpy
from commons import data_io, util_methods
from flair.data import Dictionary
from flair.trainers.language_model_trainer import TextCorpus
from torch import nn as nn

from language_modeling.flair_language_model_trainer import LanguageModelTrainer

if __name__ == '__main__':

    # path = '../data'# on hpc
    path = '..'# on gunther
    # path = '/home/tilo/hpc/data'
    # path = '/home/tilo/data'
    # data_path = path+'/ml_nlp_parsed'
    data_source = path + '/opensubtitles_de.zip'

    # path = '/home/tilo/data'
    # lines_g = (d['pdf_full_text'].replace('\n', '') for d in data_io.read_jsons_from_file('%s/arxiv.jsonl.gz' % path)
    #            if isinstance(d['pdf_full_text'],str) and len(d['pdf_full_text']) > 10)

    # lines_g = data_io.read_lines_from_files(data_path)
    lines_g = build_opensubtitles_lines_generator(data_source)

    corpus_path = '%s/de_corpus' % path
    # corpus_path = '%s/corpus' % '/tmp'
    build_train_valid_test_files = True
    if build_train_valid_test_files:
        if os.path.isdir(corpus_path):
            shutil.rmtree(corpus_path)
        if not os.path.exists(corpus_path): os.mkdir(corpus_path)
        num_train_docs = 20_000
        train_file = '%s/train.txt'% corpus_path
        data_io.write_to_file(train_file, (next(lines_g) for k in range(num_train_docs)))
        train_split_folder = corpus_path + '/train'
        if not os.path.exists(train_split_folder): os.mkdir(train_split_folder)
        train_lines_g = data_io.read_lines(train_file)
        num_train_docs = sum((1 for line in data_io.read_lines(train_file)))
        lines_per_split = 100
        num_train_splits = int(numpy.ceil(num_train_docs / lines_per_split))
        for k in range(num_train_splits):
            split_file = train_split_folder + '/train_split_%d'%k
            data_io.write_to_file(split_file, (next(train_lines_g) for k in range(lines_per_split)))

        # shutil.copy(train_file, train_split_folder + '/train_split_%d'%k)
        data_io.write_to_file('%s/valid.txt' % corpus_path, (next(lines_g) for k in range(lines_per_split)))
        data_io.write_to_file('%s/test.txt' % corpus_path, (next(lines_g) for k in range(lines_per_split)))

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
    print(language_model)
    # checkpoint = LanguageModel.load_checkpoint(Path('language_modeling/flair_resources/language_model/checkpoint.pt'))
    #
    # trainer = LanguageModelTrainer(corpus = corpus,model=checkpoint['model'],epoch=checkpoint['epoch'],split=checkpoint['split'],optimizer_state=checkpoint['optimizer_state_dict'])
    trainer = LanguageModelTrainer(corpus = corpus,model=language_model)

    path = 'flair_resources/language_model'
    trainer.train(path,checkpoint=True,
                  sequence_length=20,
                  learning_rate=20,
                  mini_batch_size=64,
                  max_epochs=10,
                  use_dataparallel=True
                  )