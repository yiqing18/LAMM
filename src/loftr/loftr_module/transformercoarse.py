import copy
import torch
import torch.nn as nn
from .linear_attention import LinearAttention, FullAttention
import numpy as np
import torch.nn.functional as F
from src.utils.plotting import *


class LoFTREncoderLayerCS(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayerCS, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead


        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)


        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )


        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):

        bs = x.size(0)
        query, key, value = x, source, source

  
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim) 
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask) 
        message = self.merge(message.view(bs, -1, self.nhead*self.dim)) 
        message = self.norm1(message)

  
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class LocalFeatureTransformerCS(nn.Module):
    def __init__(self, config):
        super(LocalFeatureTransformerCS, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = LoFTREncoderLayerCS(config['d_model'], config['nhead'], config['attention'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()
        self.k = 5
        self.conv = nn.Conv2d(256+(self.k*4),256,kernel_size=1, stride=1, padding=0, bias=False)
        self.StrucConv = nn.Conv2d(3*self.k,256,kernel_size=1, stride=1, padding=0, bias=False)
        self.merge = nn.Conv2d(256*2,256,kernel_size=1, stride=1, padding=0, bias=False)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):


        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"
        

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1
