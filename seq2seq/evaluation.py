
import torch
import torch.nn as nn

######################################################################
# Define Evaluation
# -----------------
# Greedy decoding is the decoding method that we use during training when
# we are **NOT** using teacher forcing. In other words, for each time
# step, we simply choose the word from ``decoder_output`` with the highest
# softmax value. This decoding method is optimal on a single time-step
# level.
#
# To facilite the greedy decoding operation, we define a
# ``GreedySearchDecoder`` class. When run, an object of this class takes
# an input sequence (``input_seq``) of shape *(input_seq length, 1)*, a
# scalar input length (``input_length``) tensor, and a ``max_length`` to
# bound the response sentence length. The input sentence is evaluated
# using the following computational graph:
#
# **Computation Graph:**
#
#    1) Forward input through encoder model.
#    2) Prepare encoder's final hidden layer to be first hidden input to the decoder.
#    3) Initialize decoder's first input as SOS_token.
#    4) Initialize tensors to append decoded words to.
#    5) Iteratively decode one word token at a time:
#        a) Forward pass through decoder.
#        b) Obtain most likely word token and its softmax score.
#        c) Record token and score.
#        d) Prepare current token to be next decoder input.
#    6) Return collections of word tokens and scores.
#
from seq2seq.getting_processing_data import MAX_LENGTH, indexesFromSentence, SOS_token

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):

        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token

        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)

        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)

            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)

            decoder_input = torch.unsqueeze(decoder_input, 0)
        return all_tokens, all_scores

def evaluate(searcher, voc, sentence, max_length=MAX_LENGTH):
    indexes_batch = [indexesFromSentence(voc, sentence)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    tokens, scores = searcher(input_batch, lengths, max_length)
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def interactive_chat(searcher, voc,normalizeString_fun):

    while(1):
        # try:
        input_sentence = input('> ')
        if input_sentence == 'q' or input_sentence == 'quit': break
        input_sentence = normalizeString_fun(input_sentence)
        output_words = evaluate(searcher, voc, input_sentence)
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
        print('Bot:', ''.join(output_words))

        # except KeyError:
        #     print("Error: Encountered unknown word.")
