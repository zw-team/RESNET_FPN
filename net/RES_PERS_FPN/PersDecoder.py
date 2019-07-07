import torch
import torch.nn as nn
import torch.nn.functional as functional
from net.RES_PERS_FPN.BasicPersConv2d import BasicPersConv2d

class PersDecoder(nn.Module):
    def __init__(self, IF_BN=True, **kwargs):
        super(PersDecoder, self).__init__()
        self.Decoder_Block_1_1 = BasicPersConv2d(2048, 512, 1, 1, if_Bn=IF_BN, **kwargs)
        self.Decoder_Block_1_2 = BasicPersConv2d(512, 512, 3, 1, if_Bn=IF_BN, **kwargs)

        self.Decoder_Block_2_1 = BasicPersConv2d(512, 256, 1, 1, if_Bn=IF_BN, **kwargs)
        self.Decoder_Block_2_2 = BasicPersConv2d(256, 256, 3, 1, if_Bn=IF_BN, **kwargs)

        self.Decoder_Block_3_1 = BasicPersConv2d(256, 64, 1, 1, if_Bn=IF_BN, **kwargs)
        self.Decoder_Block_3_2 = BasicPersConv2d(64, 64, 3, 1, if_Bn=IF_BN, **kwargs)
        self.Decoder_Block_3_3 = BasicPersConv2d(64, 32, 3, 1, if_Bn=IF_BN, **kwargs)
        self.output = nn.Sequential(nn.Conv2d(32, 1, 1, 1, 0), nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, B5_C3, B4_C3, B3_C3, B2_C2, pers):
        pers_maxpool_16 = functional.max_pool2d(pers, kernel_size=16, stride=16)
        pers_maxpool_8 = functional.max_pool2d(pers, kernel_size=8, stride=8)
        pers_maxpool_4 = functional.max_pool2d(pers, kernel_size=4, stride=4)
        net = self.Decoder_Block_1_1(B5_C3, pers_maxpool_16)
        net = self.Decoder_Block_1_2(net, pers_maxpool_16)
        net = functional.interpolate(
            net,
            size=B4_C3.shape[2:4],
            mode="bilinear",
            align_corners=True) + B4_C3
        net = self.Decoder_Block_2_1(net, pers_maxpool_8)
        net = self.Decoder_Block_2_2(net, pers_maxpool_8)
        net = functional.interpolate(
            net,
            size=B3_C3.shape[2:4],
            mode="bilinear",
            align_corners=True) + B3_C3
        net = self.Decoder_Block_3_1(net, pers_maxpool_4)
        net = self.Decoder_Block_3_2(net, pers_maxpool_4)
        net = self.Decoder_Block_3_3(net, pers_maxpool_4)
        net = functional.interpolate(
            net,
            size=B2_C2.shape[2:4],
            mode="bilinear",
            align_corners=True) + B2_C2
        return self.output(net)