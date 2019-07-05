
import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math

from torch.nn import DataParallel

from seq2seq.evaluation import evaluate, GreedySearchDecoder
from seq2seq.getting_processing_data import SOS_token, MAX_LENGTH
from seq2seq.seq2seq_dataprocessor import GAP
from seq2seq.seq2seq_models import EncoderRNN, LuongAttnDecoderRNN
from text_classification.classifiers.common import DataProcessorInterface

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

######################################################################
# Define Training Procedure
# -------------------------
#
# Masked loss
# ~~~~~~~~~~~
#
# Since we are dealing with batches of padded sequences, we cannot simply
# consider all elements of the tensor when calculating loss. We define
# ``maskNLLLoss`` to calculate our loss based on our decoder’s output
# tensor, the target tensor, and a binary mask tensor describing the
# padding of the target tensor. This loss function calculates the average
# negative log likelihood of the elements that correspond to a *1* in the
# mask tensor.
#

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


######################################################################
# Single training iteration
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The ``train`` function contains the algorithm for a single training
# iteration (a single batch of inputs).
#
# We will use a couple of clever tricks to aid in convergence:
#
# -  The first trick is using **teacher forcing**. This means that at some
#    probability, set by ``teacher_forcing_ratio``, we use the current
#    target word as the decoder’s next input rather than using the
#    decoder’s current guess. This technique acts as training wheels for
#    the decoder, aiding in more efficient training. However, teacher
#    forcing can lead to model instability during inference, as the
#    decoder may not have a sufficient chance to truly craft its own
#    output sequences during training. Thus, we must be mindful of how we
#    are setting the ``teacher_forcing_ratio``, and not be fooled by fast
#    convergence.
#
# -  The second trick that we implement is **gradient clipping**. This is
#    a commonly used technique for countering the “exploding gradient”
#    problem. In essence, by clipping or thresholding gradients to a
#    maximum value, we prevent the gradients from growing exponentially
#    and either overflow (NaN), or overshoot steep cliffs in the cost
#    function.
#
# .. figure:: /_static/img/chatbot/grad_clip.png
#    :align: center
#    :width: 60%
#    :alt: grad_clip
#
# Image source: Goodfellow et al. *Deep Learning*. 2016. https://www.deeplearningbook.org/
#
# **Sequence of Operations:**
#
#    1) Forward pass entire input batch through encoder.
#    2) Initialize decoder inputs as SOS_token, and hidden state as the encoder's final hidden state.
#    3) Forward input batch sequence through decoder one time step at a time.
#    4) If teacher forcing: set next decoder input as the current target; else: set next decoder input as current decoder output.
#    5) Calculate and accumulate loss.
#    6) Perform backpropagation.
#    7) Clip gradients.
#    8) Update encoder and decoder model parameters.
#
#
# .. Note ::
#
#   PyTorch’s RNN modules (``RNN``, ``LSTM``, ``GRU``) can be used like any
#   other non-recurrent layers by simply passing them the entire input
#   sequence (or batch of sequences). We use the ``GRU`` layer like this in
#   the ``encoder``. The reality is that under the hood, there is an
#   iterative process looping over each time step calculating hidden states.
#   Alternatively, you ran run these modules one time-step at a time. In
#   this case, we manually loop over the sequences during the training
#   process like we must do for the ``decoder`` model. As long as you
#   maintain the correct conceptual model of these modules, implementing
#   sequential models can be very straightforward.
#
#
teacher_forcing_ratio = .9

def trainIters(model_name, dp:DataProcessorInterface,data_path,encoder:EncoderRNN, decoder:LuongAttnDecoderRNN, save_dir,
               corpus_name,
               batch_size=64,
               clip = 50.0,
               learning_rate = 0.0001,
               decoder_learning_ratio = 5.0,
               n_iteration = 4000,
               print_every = 1,
               save_every = 500,
                encoder_optimizer_sd=None,
                decoder_optimizer_sd=None,
               ):

    encoder.train()
    decoder.train()

    encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

    print_loss = 0
    if encoder_optimizer_sd is not None:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    if decoder_optimizer_sd is not None:
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    encoder_parallel = DataParallel(encoder,dim=1)
    decoder_parallel = DataParallel(decoder,dim=1)
    def train_on_batch(training_batch):
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_variable = input_variable.to(device)
        lengths = lengths.to(device)
        target_variable = target_variable.to(device)
        mask = mask.to(device)

        loss = 0
        print_losses = []
        n_totals = 0

        encoder_outputs, encoder_hidden = encoder_parallel(input_variable, lengths.unsqueeze(0))

        decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
        decoder_input = decoder_input.to(device)

        decoder_hidden = encoder_hidden[:decoder.n_layers]

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder_parallel(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                decoder_output = decoder_output.permute(1,0)
                decoder_input = target_variable[t].view(1, -1)
                mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder_parallel(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                decoder_output = decoder_output.permute(1,0)
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
                decoder_input = decoder_input.to(device)
                mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal

        loss.backward()

        _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

        encoder_optimizer.step()
        decoder_optimizer.step()

        return sum(print_losses) / n_totals

    get_batch_fun = dp.build_get_batch_fun(data_path,batch_size=batch_size)

    for iteration in range(n_iteration + 1):
        training_batch = get_batch_fun()

        loss = train_on_batch(training_batch)
        print_loss += loss

        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0
            encoder.eval()
            decoder.eval()
            searcher = GreedySearchDecoder(encoder, decoder)
            to_print=[]
            for input_sentence in ['The Embedd%strix'%GAP, 'evaluati%sric'%GAP,'Dee%sarning'%GAP,'Natu%suage'%GAP,'The%sen'%GAP]:
                input_batch, lengths = dp.transform([input_sentence])
                input_batch = input_batch.to(device)
                lengths = lengths.to(device)

                tokens, scores = searcher(input_batch, lengths, MAX_LENGTH)
                decoded_words = [dp.voc.index2word[token.item()] for token in tokens]
                decoded_words = [x for x in decoded_words if not (x == 'EOS' or x == 'PAD')]
                to_print.append('Input: %s; filled: %s' % (input_sentence, input_sentence.replace(GAP, ''.join(decoded_words))))
            print('|'.join(to_print))

            encoder.train()
            decoder.train()

        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder.n_layers, decoder.n_layers, encoder.hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': dp.voc.__dict__,
                'embedding': encoder.embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format('model', 'checkpoint')))
