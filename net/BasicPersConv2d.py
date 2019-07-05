import torch
import torch.nn as nn
from op_wrapper.pad_conv2d_wrapper import BasicPerspectiveDilatedConv2D

class BasicPersConv2d(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride, 
                 pad=0, 
                 if_Bn=True,
                 if_Bias=True,
                 activation=nn.ReLU(inplace=True)):
        super(BasicPersConv2d, self).__init__()
        self.kernel_size = kernel_size
        if kernel_size == 1:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad)
        else:
            self.conv2d = BasicPerspectiveDilatedConv2D(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.if_Bn = if_Bn
        if self.if_Bn:
            self.Bn = nn.BatchNorm2d(out_channels)
        self.activation = activation
    
    def forward(self, x, pers):
        x = self.conv2d(x) if self.kernel_size == 1 else self.conv2d(x, pers)
        if self.if_Bn:
            x = self.Bn(x)
        if not(self.activation == None):
            x = self.activation(x)
        return x