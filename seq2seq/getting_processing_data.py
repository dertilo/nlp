from commons import data_io
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

######################################################################
# Preparations
# ------------
#
# To start, Download the data ZIP file
# `here <https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html>`__
# and put in a ``data/`` directory under the current directory.
#
# After that, let’s import some necessities.

######################################################################
# Load & Preprocess Data
# ----------------------
#
# The next step is to reformat our data file and load the data into
# structures that we can work with.
#
# The `Cornell Movie-Dialogs
# Corpus <https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html>`__
# is a rich dataset of movie character dialog:
#
# -  220,579 conversational exchanges between 10,292 pairs of movie
#    characters
# -  9,035 characters from 617 movies
# -  304,713 total utterances
#
# This dataset is large TODO(tilo):"large"? thefaq! this tiny!
# and diverse, and there is a great variation of
# language formality, time periods, sentiment, etc. Our hope is that this
# diversity makes our model robust to many forms of inputs and queries.

base_path = 'chatterbot'
corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join(base_path+"/data", corpus_name)
save_dir = os.path.join(base_path+"/data", "save")


def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

printLines(os.path.join(corpus, "movie_lines.txt"))


def loadLines(fileName, fields):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines


# Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            lineIds = eval(convObj["utteranceIDs"])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations


# Extracts pairs of sentences from conversations
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # End-of-sentence token

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS",UNK_token:'#'}
        self.word2index = {word:idx for idx,word in self.index2word.items()}
        self.num_words = 4  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS",UNK_token:'#'}
        self.word2index = {word:idx for idx,word in self.index2word.items()}
        self.num_words = 4 # Count default tokens

        for word in keep_words:
            self.addWord(word)

MAX_LENGTH = 10  # Maximum sentence length to consider

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def readVocs(datafile, corpus_name):
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')

    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def batch2TrainData(voc:Voc, pair_batch,tokenize_fun=lambda x:x.split(' ')):
    pair_batch.sort(key=lambda x: len(tokenize_fun(x[0])), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc,tokenize_fun)
    output, mask, max_target_len = outputVar(output_batch, voc,tokenize_fun)
    return inp, lengths, output, mask, max_target_len

def loadPrepareData(corpus_name = "cornell movie-dialogs corpus",
                    datafile = os.path.join(corpus, "formatted_movie_lines.txt")
                    ):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    # Print some pairs to validate
    print("\npairs:")
    for pair in pairs[:10]:
        print(pair)

    pairs = trimRareWords(voc, pairs, MIN_COUNT)
    def batch2TrainData_fun(voc,pair_batch):
        return batch2TrainData(voc,pair_batch)
    return voc, pairs,batch2TrainData_fun,normalizeString

def load_prepare_keyword_sentence_data(
        datafile:str,
        limit=100
                    ):
    corpus_name = "sci-keyword-sentencs"
    data_g = data_io.read_jsons_from_file(datafile,limit=100)
    voc = Voc(corpus_name)
    [voc.addWord(char) for d in data_g for char in d['sentence']]
    voc.trim(10)
    assert '#' in voc.word2index.keys()
    print("Counted words:", voc.num_words)
    # def infinite_data_g():
    #     while True:
    #         for d in data_io.read_jsons_from_file(datafile):
    #             yield d

    def normalizeString(s):
        s = re.sub(r"\s+", r" ", s).strip()
        s = ''.join([c if c in voc.word2index else voc.index2word[UNK_token] for c in s])
        return s

    pairs = [(normalizeString(' '.join(d['keywords'])),normalizeString(d['sentence'])) for d in data_io.read_jsons_from_file(datafile,limit=limit)]
    print('got %d train-samples'%len(pairs))
    def batch2TrainData_fun(voc,pair_batch):
        return batch2TrainData(voc,pair_batch,lambda s:list(s))

    return voc, pairs,batch2TrainData_fun,normalizeString

MIN_COUNT = 3    # Minimum word count threshold for trimming

def trimRareWords(voc, pairs, MIN_COUNT):
    voc.trim(MIN_COUNT)
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

def indexesFromSentence(voc, sentence,tokenize_fun=lambda x:list(x)):
    return [voc.word2index[word] for word in tokenize_fun(sentence)] + [EOS_token]

# def indexesFromSentence(voc, sentence,tokenize_fun=lambda x:x.split(' ')):
#     return [voc.word2index[word] for word in tokenize_fun(sentence)] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def inputVar(l, voc,tokenize_fun):
    indexes_batch = [indexesFromSentence(voc, sentence,tokenize_fun) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

def outputVar(l, voc,tokenize_fun):
    indexes_batch = [indexesFromSentence(voc, sentence,tokenize_fun) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


if __name__ == '__main__':
    # voc, pairs = loadPrepareData()
    load_prepare_keyword_sentence_data('/tmp/keywords_to_sentence.csv')
    print()
