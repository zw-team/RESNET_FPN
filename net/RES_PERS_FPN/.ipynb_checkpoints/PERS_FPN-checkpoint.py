import torch
import torch.nn as nn
import torch.nn.functional as functional
from net.RES_PERS_FPN.Encoder import Encoder
from net.RES_PERS_FPN.PersDecoder import PersDecoder
from op_wrapper.adaptive_sigmoid_wrapper import AdaptiveSigmoid
from op_wrapper.pad_conv2d_wrapper import PerspectiveDilatedConv2dLayer

class PERS_FPN(nn.Module):
    def __init__(self, pretrain=True, IF_BN=True, **kwargs):
        super(PERS_FPN, self).__init__()
        self.encoder = Encoder(pretrain=pretrain)
        self.decoder = PersDecoder(IF_BN=True, **kwargs)
        self.ada_sig_params = []
        self.conv_params = [*list(self.encoder.parameters())]
        for m in self.decoder.modules():
            if isinstance(m, AdaptiveSigmoid):
                self.ada_sig_params.append(m.params)
            elif isinstance(m, nn.Conv2d):
                self.conv_params.append(m.weight)
                self.conv_params.append(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                self.conv_params.append(m.weight)
                self.conv_params.append(m.bias)
            elif isinstance(m, PerspectiveDilatedConv2dLayer):
                self.conv_params.append(m.weight)
                self.conv_params.append(m.bias)

    def forward(self, x, pers):
        B5_C3, B4_C3, B3_C3, B2_C2 = self.encoder(x)
        output = self.decoder(B5_C3, B4_C3, B3_C3, B2_C2, pers)
        return output
    
    def get_params(self):
        return self.conv_params, self.ada_sig_params