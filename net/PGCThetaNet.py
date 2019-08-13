import torch.nn as nn
from torchvision import models
import torch.nn.functional as functional
import torch
import numpy as np
from collections import OrderedDict
from op_wrapper.gaussian_smooth_wrapper import BasicGaussianSmoothFunctor, GaussianSmoothFunction
from op_wrapper.dilated_conv2d_wrapper import BasicDilatedConv2D, DilatedConv2dLayer
from op_wrapper.adaptive_sigmoid_wrapper import AdaptiveSigmoid

class BasicPGCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, gaussian_kernel_size, **kwargs):
        super(BasicPGCBlock, self).__init__()
        self.conv2d = BasicDilatedConv2D(in_channels, out_channels, kernel_size=kernel_size, padding=kwargs['atrous'], dilation=kwargs['atrous'])
        self.gau_smooth = BasicGaussianSmoothFunctor(gaussian_kernel_size, sigma=kwargs['sigma_args'], updates_signal=[True, True, True, True])
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x, perspective_map):
        x = self.gau_smooth(x, perspective_map)
        x = self.conv2d(x)
        x = self.relu(x)
        return x
        
class Backend(nn.Module):
    def __init__(self, in_channels, gaussian_kernel_size, gs_layers=[], **kwargs):
        super(Backend, self).__init__()
        self.MaxPooling = nn.MaxPool2d(8)
        channels_setting = [512, 512, 512, 256, 128, 64]
        self.Block_List = torch.nn.ModuleList()
        v = in_channels
        self.gs_layers = gs_layers
        for i in range(len(channels_setting)):
            if not(i in self.gs_layers):
                self.Block_List.append(nn.Sequential(BasicDilatedConv2D(v, channels_setting[i], 3, padding=kwargs['atrous'], dilation=kwargs['atrous']), nn.ReLU(inplace=True)))
            else:
                self.Block_List.append(BasicPGCBlock(v, channels_setting[i], 3, gaussian_kernel_size, **kwargs))
            v = channels_setting[i]
    
    def forward(self, x, perspective_map):
        perspective_map = self.MaxPooling(perspective_map)
        for i in range(len(self.Block_List)):
            layer = self.Block_List[i]
            if not(i in self.gs_layers):
                x = layer(x)
            else:
                x = layer(x, perspective_map)
        return x

    
class PGCThetaNet(nn.Module):
    def __init__(self, load_path=None, gaussian_size=7, gs_layers=[], **kwargs):
        super(PGCThetaNet, self).__init__()
        self.front_end = nn.Sequential(*(list(list(models.vgg16(True).children())[0].children())[0:23]))
        self.back_end = Backend(512, gaussian_size, gs_layers, **kwargs)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        if not(load_path == None):
            new_state_dict = OrderedDict()
            state_dict = torch.load(load_path)
            count = 0
            for k,v in state_dict.items():
                if 'back_end' in k:
                    if count not in gs_layers:
                        name_prefix = "back_end.Block_List." + str(count) + ".0"
                        if 'weight' in k:
                            new_state_dict[name_prefix + '.dilated_conv2d.weight'] = v
                        elif 'bias' in k:
                            new_state_dict[name_prefix + '.dilated_conv2d.bias'] = v
                            count += 1
                    else:
                        name_prefix = "back_end.Block_List." + str(count) 
                        if 'weight' in k:
                            new_state_dict[name_prefix + '.conv2d.dilated_conv2d.weight'] = v
                        elif 'bias' in k:
                            new_state_dict[name_prefix + '.conv2d.dilated_conv2d.bias'] = v
                            new_state_dict[name_prefix + '.gau_smooth.sigma_map_generator.params'] = torch.FloatTensor(kwargs['sigma_args'])
                            count += 1
                    
                else:
                    new_state_dict[k] = v
            self.load_state_dict(new_state_dict)
            
        else:
            for m in self.output_layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, perspective_map):
        img_shape = x.shape
        front_end = self.front_end(x)
        back_end = self.back_end(front_end, perspective_map)
        output = self.output_layer(back_end)
        return output

    def get_params(self):
        self.ada_sig_params = []
        self.conv_params = []
        for m in self.modules():
            if isinstance(m, AdaptiveSigmoid):
                self.ada_sig_params.append(m.params)
            elif isinstance(m, DilatedConv2dLayer):
                self.conv_params.append(m.weight)
                self.conv_params.append(m.bias)
        return self.conv_params, self.ada_sig_params