# -*- coding: utf-8 -*-
######################################################################
# **Acknowledgements**
#
# This tutorial borrows code from the following sources:
#
# 1) Yuan-Kuei Wu’s pytorch-chatbot implementation:
#    https://github.com/ywk991112/pytorch-chatbot
#
# 2) Sean Robertson’s practical-pytorch seq2seq-translation example:
#    https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation
#
# 3) FloydHub’s Cornell Movie Corpus preprocessing code:
#    https://github.com/floydhub/textutil-preprocess-cornell-movie-corpus
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
sys.path.append('.')
from scipy.sparse import csr_matrix #TODO(tilo): if not imported before torch it throws: ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found
from seq2seq.getting_processing_data import Voc

from seq2seq.seq2seq_dataprocessor import Seq2GapDataProcessor
from seq2seq.evaluation import GreedySearchDecoder, interactive_chat
from seq2seq.seq2seq_models import EncoderRNN, LuongAttnDecoderRNN
from seq2seq.seq2seq_training import trainIters
import torch
import torch.nn as nn
from torch import optim
import random

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# voc,pairs,process_batch_fun,normalizeString_fun = loadPrepareData()
# datafile = '../data/keywords_to_sentence.jsonl.gz'
# datafile = '/tmp/keywords_to_sentence.jsonl.gz'
# voc,pairs,process_batch_fun,normalizeString_fun = load_prepare_keyword_sentence_data(datafile, limit=100000000)


attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 1024
# hidden_size = 50
encoder_n_layers = 1
decoder_n_layers = 1
dropout = 0.1
batch_size = 64
embedding_size=50

loadFilename = None

# data_path = '/home/tilo/hpc/data/ml_nlp_parsed'
data_path = '../data/ml_nlp_parsed'


def prepare_model(loadFilename):
    dp = Seq2GapDataProcessor()

    if loadFilename:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename)
        # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        embedding_sd = checkpoint['embedding']
        dp.voc.__dict__ = checkpoint['voc_dict']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']

    else:
        checkpoint = None
        dp.fit(data_path)
        encoder_optimizer_sd,decoder_optimizer_sd = None,None

    print('Building encoder and decoder ...')
    embedding = nn.Embedding(dp.voc.num_words, embedding_size)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)

    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, dp.voc.num_words, decoder_n_layers, dropout)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    return encoder,decoder,checkpoint,encoder_optimizer_sd,decoder_optimizer_sd, dp


encoder,decoder,checkpoint,encoder_optimizer_sd,decoder_optimizer_sd, dp = prepare_model(loadFilename)

print(encoder)
print(decoder)
save_dir = 'seq2seq_models'
model_name = 'gapfiller'
trainIters(model_name, dp, data_path, encoder, decoder,
           save_dir,
           learning_rate=0.001,
           decoder_learning_ratio=1.0,
           corpus_name='arxiv_sentences',
           n_iteration=200000,
           print_every=200,
           encoder_optimizer_sd=encoder_optimizer_sd,
           decoder_optimizer_sd=decoder_optimizer_sd
           )

encoder,decoder,checkpoint,encoder_optimizer_sd,decoder_optimizer_sd, dp = prepare_model('seq2seq_models/%s/arxiv_sentences/1-1_%d/model_checkpoint.tar'%(model_name,hidden_size))

encoder.eval()
decoder.eval()
searcher = GreedySearchDecoder(encoder, decoder)

interactive_chat(searcher, dp)
