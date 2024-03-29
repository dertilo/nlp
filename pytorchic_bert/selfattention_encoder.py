# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Transformer Model Classes & Config Class """

import math
import json
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorchic_bert.utils import split_last, merge_last


class BertConfig(NamedTuple):
    '''
    Configuration for BERT model"
    BERT: Bidirectional Encoder Representations from Transformers
    '''

    vocab_size: int = None # Size of Vocabulary
    dim: int = 768 # Dimension of Hidden Layer in Transformer Encoder
    n_layers: int = 12 # Numher of Hidden Layers
    n_heads: int = 12 # Numher of Heads in Multi-Headed Attention Layers
    dim_ff: int = 768*4 # Dimension of Intermediate Layers in Positionwise Feedforward Net
    #activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    p_drop_hidden: float = 0.1 # Probability of Dropout of various Hidden Layers
    p_drop_attn: float = 0.1 # Probability of Dropout of Attention Layers
    max_len: int = 512 # Maximum Length for Positional Embeddings
    n_segments: int = 2 # Number of Sentence Segments

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, dim, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."
    def __init__(self, vocab_size,dim,max_len,n_segments,p_drop_hidden):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, dim) # token embedding
        self.pos_embed = nn.Embedding(max_len, dim) # position embedding
        self.seg_embed = nn.Embedding(n_segments, dim) # segment(token type) embedding

        self.norm = LayerNorm(dim)
        self.drop = nn.Dropout(p_drop_hidden)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x) # (S,) -> (B, S)

        e = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.drop(self.norm(e))


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, dim,n_heads, p_drop_attn):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(p_drop_attn)
        self.scores = None # for visualization
        self.n_heads = n_heads

    def forward(self, x, mask=None):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, dim,dim_ff):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim_ff)
        self.fc2 = nn.Linear(dim_ff, dim)
        #self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(nn.Module):

    def __init__(self, dim,n_heads,dim_ff,p_drop_attn=0.1,p_drop_hidden=0.1):
        super().__init__()
        assert dim%n_heads==0
        self.attn = MultiHeadedSelfAttention(dim,n_heads,p_drop_attn)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = LayerNorm(dim)
        self.pwff = PositionWiseFeedForward(dim,dim_ff)
        self.norm2 = LayerNorm(dim)
        self.drop = nn.Dropout(p_drop_hidden)

    def forward(self, x, mask=None):
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h


class EncoderStack(nn.Module):
    
    def __init__(self, cfg:BertConfig):
        super().__init__()
        self.embed = Embeddings(cfg.vocab_size,cfg.dim,cfg.max_len,cfg.n_segments,cfg.p_drop_hidden)
        self.layers = nn.ModuleList([EncoderLayer(cfg.dim,cfg.n_heads,cfg.dim_ff,cfg.p_drop_attn,cfg.p_drop_hidden) for _ in range(cfg.n_layers)])

    def forward(self, input_ids, segment_ids, input_mask):
        h = self.embed(input_ids, segment_ids)
        for layer in self.layers:
            h = layer(h, input_mask)
        return h

