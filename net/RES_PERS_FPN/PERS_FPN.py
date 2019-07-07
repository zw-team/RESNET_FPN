import torch
import torch.nn as nn
import torch.nn.functional as functional
from net.RES_PERS_FPN.Encoder import Encoder
from net.RES_PERS_FPN.PersDecoder import PersDecoder


class PERS_FPN(nn.Module):
    def __init__(self, pretrain=True, IF_BN=True, **kwargs):
        super(PERS_FPN, self).__init__()
        self.encoder = Encoder(pretrain=pretrain)
        self.decoder = PersDecoder(IF_BN=True, **kwargs)

    def forward(self, x, pers):
        B5_C3, B4_C3, B3_C3, B2_C2 = self.encoder(x)
        output = self.decoder(B5_C3, B4_C3, B3_C3, B2_C2, pers)
        return output