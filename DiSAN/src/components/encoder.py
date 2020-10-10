import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import time
from torch.autograd import Variable
import pdb
from src.components.utils import *


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask, mask_bi):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask, mask_bi)
        return self.norm(x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask, mask_bi):
        # pdb.set_trace()
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask, mask_bi))
        return self.sublayer[1](x, self.feed_forward)
