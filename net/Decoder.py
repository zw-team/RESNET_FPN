import torch
import torch.nn as nn
import torch.nn.functional as functional
from net.BasicConv2d import BasicConv2d

class Decoder(nn.Module):
    def __init__(self, IF_BN=True, **kwargs):
        super(Decoder, self).__init__()
        self.Decoder_Block_1 = nn.Sequential(
            BasicConv2d(2048, 512, 1, 1, 0, if_Bn=IF_BN),
            BasicConv2d(512, 512, 3, 1, 1, if_Bn=IF_BN))

        self.Decoder_Block_2 = nn.Sequential(
            BasicConv2d(512, 256, 1, 1, 0, if_Bn=IF_BN),
            BasicConv2d(256, 256, 3, 1, 1, if_Bn=IF_BN))

        self.Decoder_Block_3 = nn.Sequential(
            BasicConv2d(256, 64, 1, 1, 0, if_Bn=IF_BN),
            BasicConv2d(64, 64, 3, 1, 1, if_Bn=IF_BN),
            BasicConv2d(64, 32, 3, 1, 1, if_Bn=IF_BN),
        )
        self.output = nn.Sequential(
            nn.Conv2d(32, 1, 1, 1, 0), nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, B5_C3, B4_C3, B3_C3, B2_C2):
        link_1 = functional.interpolate(
            self.Decoder_Block_1(B5_C3),
            size=B4_C3.shape[2:4],
            mode="bilinear",
            align_corners=True) + B4_C3
        link_2 = functional.interpolate(
            self.Decoder_Block_2(link_1),
            size=B3_C3.shape[2:4],
            mode="bilinear",
            align_corners=True) + B3_C3
        link_3 = functional.interpolate(
            self.Decoder_Block_3(link_2),
            size=B2_C2.shape[2:4],
            mode="bilinear",
            align_corners=True) + B2_C2

        return self.output(link_3)