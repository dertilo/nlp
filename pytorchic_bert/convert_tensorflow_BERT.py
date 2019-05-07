import sys
sys.path.append('.')
import os
from typing import NamedTuple

import torch

from pytorchic_bert.checkpoint import load_model
from pytorchic_bert.selfattention_encoder import EncoderStack

class BertConfig(NamedTuple):
    "Configuration for BERT model"
    dim: int = 768 # Dimension of Hidden Layer in Transformer Encoder
    dim_ff: int = 768*4 # Dimension of Intermediate Layers in Positionwise Feedforward Net
    n_layers: int = 12 # Numher of Hidden Layers
    p_drop_attn: float = 0.1 # Probability of Dropout of Attention Layers
    n_heads: int = 12 # Numher of Heads in Multi-Headed Attention Layers
    p_drop_hidden: float = 0.1 # Probability of Dropout of various Hidden Layers
    max_len: int = 512 # Maximum Length for Positional Embeddings
    n_segments: int = 2 # Number of Sentence Segments
    vocab_size: int = 30522 # Size of Vocabulary

if __name__ == '__main__':
    model = EncoderStack(BertConfig())
    from pathlib import Path
    home = str(Path.home())
    save_dir = home+'/data/models/uncased_L-12_H-768_A-12'
    pretrain_file = save_dir + '/bert_model.ckpt'

    load_model(model,pretrain_file)

    save_file = save_dir + '/bert_encoder_pytorch.pt'
    torch.save(model.state_dict(), save_file)
    # model.load_state_dict(
    #     torch.load(save_file, map_location=None if torch.cuda.is_available() else 'cpu'))
