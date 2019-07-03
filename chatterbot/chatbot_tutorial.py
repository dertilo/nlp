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

from chatterbot.evaluation import GreedySearchDecoder, interactive_chat
from chatterbot.models import EncoderRNN, LuongAttnDecoderRNN
from chatterbot.training import trainIters
import torch
import torch.nn as nn
from torch import optim
import random

from chatterbot.getting_processing_data import loadPrepareData, batch2TrainData, corpus_name, \
    save_dir

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

voc,pairs = loadPrepareData()

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
        encoder_optimizer_sd=None
        decoder_optimizer_sd=None
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
    return encoder,decoder,checkpoint,encoder_optimizer_sd, decoder_optimizer_sd, embedding


encoder,decoder,checkpoint,encoder_optimizer_sd, decoder_optimizer_sd, embedding = prepare_model(loadFilename,voc)

trainIters(model_name, voc, pairs, encoder, decoder, embedding,
           save_dir,
           corpus_name=corpus_name,
           n_iteration=4000,
           loadFilename=loadFilename,checkpoint=checkpoint
           )

encoder.eval()
decoder.eval()

searcher = GreedySearchDecoder(encoder, decoder)

interactive_chat(encoder, decoder, searcher, voc)
