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

from seq2seq.evaluation import GreedySearchDecoder, interactive_chat
from seq2seq.models import EncoderRNN, LuongAttnDecoderRNN
from seq2seq.training import trainIters
import torch
import torch.nn as nn
from torch import optim
import random

from seq2seq.getting_processing_data import loadPrepareData, batch2TrainData, corpus_name, \
    save_dir, load_prepare_keyword_sentence_data

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# voc,pairs,process_batch_fun,normalizeString_fun = loadPrepareData()
voc,pairs,process_batch_fun,normalizeString_fun = load_prepare_keyword_sentence_data('../data/keywords_to_sentence.jsonl.gz',limit=10000000)

model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

loadFilename = None

def prepare_model(loadFilename,voc):

    if loadFilename:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']
    else:
        checkpoint = None

    print('Building encoder and decoder ...')
    embedding = nn.Embedding(voc.num_words, hidden_size)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)

    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    return encoder,decoder,checkpoint


encoder,decoder,checkpoint = prepare_model(loadFilename,voc)

trainIters(model_name, voc, pairs,process_batch_fun, normalizeString_fun,encoder, decoder,
           save_dir,
           corpus_name=corpus_name,
           n_iteration=5000,
           print_every=100,
           loadFilename=loadFilename,
           checkpoint=checkpoint
           )

encoder.eval()
decoder.eval()
searcher = GreedySearchDecoder(encoder, decoder)

interactive_chat(searcher, voc,normalizeString_fun)