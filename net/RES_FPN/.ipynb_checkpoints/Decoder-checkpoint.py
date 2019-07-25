import torch
import torch.nn as nn
import torch.nn.functional as functional
from net.RES_FPN.BasicConv2d import BasicConv2d

class Decoder(nn.Module):
    def __init__(self, IF_BN=True, **kwargs):
        super(Decoder, self).__init__()
        self.Decoder_Block_1_1 = nn.Conv2d(2048, 512, 1, 1, 0)
        self.Decoder_Block_1_2 = nn.Sequential(BasicConv2d(2048, 512, 3, 1, 1, if_Bn=IF_BN), 
                                               BasicConv2d(512, 512, 3, 1, 1, if_Bn=IF_BN))
        
        self.Decoder_Block_2_1 = nn.Conv2d(512, 256, 1, 1, 0)
        self.Decoder_Block_2_2 = nn.Sequential(BasicConv2d(512, 256, 3, 1, 1, if_Bn=IF_BN), 
                                               BasicConv2d(256, 256, 3, 1, 1, if_Bn=IF_BN))

        self.Decoder_Block_3_1 = nn.Conv2d(256, 64, 1, 1, 0)
        self.Decoder_Block_3_2 = nn.Sequential(BasicConv2d(256, 64, 3, 1, 1, if_Bn=IF_BN), 
                                               BasicConv2d(64, 64, 3, 1, 1, if_Bn=IF_BN))
        
        self.output = nn.Sequential(nn.Conv2d(64, 1, 1, 1, 0), nn.LeakyReLU(negative_slope=0.01, inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, B5_C3, B4_C3, B3_C3, B2_C2):
        net = self.Decoder_Block_1_1(B5_C3) + self.Decoder_Block_1_2(B5_C3)
        net = functional.interpolate(
            net,
            size=B4_C3.shape[2:4],
            mode="bilinear",
            align_corners=True) + B4_C3
        net = self.Decoder_Block_2_1(net) + self.Decoder_Block_2_2(net)
        net = functional.interpolate(
            net,
            size=B3_C3.shape[2:4],
            mode="bilinear",
            align_corners=True) + B3_C3
        net = self.Decoder_Block_3_1(net) + self.Decoder_Block_3_2(net)
        net = functional.interpolate(
            net,
            size=B2_C2.shape[2:4],
            mode="bilinear",
            align_corners=True) + B2_C2
        return self.output(net)
    
    def mapping(self, x, mode='identity'):
        if mode == 'identity':
            return x
        else:
            raise NameError("Not support such mapping!")