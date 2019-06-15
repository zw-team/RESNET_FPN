import torch
import torch.nn as nn
import torch.nn.functional as functional
from net.BasicConv2d import BasicConv2d


class Bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = BasicConv2d(
            in_channels,
            mid_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            pad=(0, 0),
            if_Bn=True,
            if_Bias=False,
            activation=None)
        self.conv2 = BasicConv2d(
            mid_channels,
            mid_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            pad=(1, 1),
            if_Bn=True,
            if_Bias=False,
            activation=None)
        self.conv3 = BasicConv2d(
            mid_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            pad=(0, 0),
            if_Bn=True,
            if_Bias=False)
        if not (in_channels == out_channels):
            self.downsample = BasicConv2d(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=(2, 2),
                pad=(0, 0),
                if_Bn=True,
                if_Bias=False,
                activation=None)

    def forward(self, x):
        x_res = self.conv1(x)
        x_res = self.conv2(x_res)
        x_res = self.conv3(x_res)
        if self.in_channels == self.out_channels:
            x_ori = x
        else:
            x_ori = self.downsample(x)
        return x_res + x_ori