import torch
import torch.nn as nn
import torch.nn.functional as functional
from net.Encoder import Encoder
from net.BasicConv2d import BasicConv2d
from net.Decoder import Decoder


class FPN(nn.Module):
    def __init__(self, pretrain=True, IF_BN=True, **kwargs):
        super(FPN, self).__init__()
        self.encoder = Encoder(pretrain=pretrain)
        self.decoder = Decoder(IF_BN=True, **kwargs)

    def forward(self, x):
        B5_C3, B4_C3, B3_C3, B2_C2 = self.encoder(x)
        output = self.decoder(B5_C3, B4_C3, B3_C3, B2_C2)
        return output