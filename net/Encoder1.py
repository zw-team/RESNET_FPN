import torch
import torch.nn as nn
from net.BasicConv2d import BasicConv2d

class Encoder(nn.Module):
    def __init__(self, pretrain=True, IF_BN=True, **kwargs):
        super(Encoder, self).__init__()
        self.B1_C2 = nn.Sequential(
            BasicConv2d(3, 64, 3, 1, 1, if_Bn=IF_BN), 
            BasicConv2d(64, 64, 3, 1, 1, if_Bn=IF_BN)
        )
        self.B2_C2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConv2d(64, 128, 3, 1, 1, if_Bn=IF_BN), 
            BasicConv2d(128, 128, 3, 1, 1, if_Bn=IF_BN)
        )
        self.B3_C3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConv2d(128, 256, 3, 1, 1, if_Bn=IF_BN), 
            BasicConv2d(256, 256, 3, 1, 1, if_Bn=IF_BN),
            BasicConv2d(256, 256, 3, 1, 1, if_Bn=IF_BN)
        )
        self.B4_C3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConv2d(256, 512, 3, 1, 1, if_Bn=IF_BN), 
            BasicConv2d(512, 512, 3, 1, 1, if_Bn=IF_BN),
            BasicConv2d(512, 512, 3, 1, 1, if_Bn=IF_BN)
        )
        self.B5_C3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConv2d(512, 512, 3, 1, 1, if_Bn=IF_BN), 
            BasicConv2d(512, 512, 3, 1, 1, if_Bn=IF_BN),
            BasicConv2d(512, 512, 3, 1, 1, if_Bn=IF_BN)
        )
        if pretrain == False:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            if IF_BN:
                self.load_state_dict(
                    torch.load(
                        "/home/zzn/PycharmProjects/MagNet/StateDicts/vggbn16_10conv2d_statedict.pkl"
                    )
                )
            else:
                self.load_state_dict(
                    torch.load(
                        "/home/zzn/PycharmProjects/MagNet/StateDicts/vgg16_10conv2d_statedict.pkl"
                    )
                )
                             
    def forward(self, x):
        B1_C2_output = self.B1_C2(x)
        B2_C2_output = self.B2_C2(B1_C2_output)
        B3_C3_output = self.B3_C3(B2_C2_output)
        B4_C3_output = self.B4_C3(B3_C3_output)
        B5_C3_output = self.B5_C3(B4_C3_output)
        return B5_C3_output, B4_C3_output, B3_C3_output,  B2_C2_output, B1_C2_output