import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import Module
from op_wrapper.adaptive_sigmoid_wrapper import AdaptiveSigmoid
import gaussian_perspective_smooth as gaussian_smooth

class GaussianSmoothFunction(Function):
    @staticmethod
    def forward(ctx, *args):
        if len(args) != 4:
            print("wrong input parameters number, check the input")
            return
        input = args[0]
        sigma_map = args[1]
        ctx.kernel_h = args[2]
        ctx.kernel_w = args[3]
        output = gaussian_smooth.forward(input, sigma_map, ctx.kernel_h, ctx.kernel_w)
        ctx.save_for_backward(input, sigma_map)
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        if len(grad_outputs) != 1:
            print("Wrong output number, check your output")
            return
        input, sigma_map = ctx.saved_tensors
        grad_copy = grad_outputs[0].clone()
        grad_input, grad_sigma_map = gaussian_smooth.backward(input, sigma_map, grad_copy, ctx.kernel_h, ctx.kernel_w)
        return grad_input, grad_sigma_map, None, None

    
class BasicGaussianSmoothFunctor(Module):
    def __init__(self, kernel_size, **kwargs):
        super(BasicGaussianSmoothFunctor, self).__init__()
        self.sigma_map_generator = AdaptiveSigmoid(**kwargs)
#         self.sigma_map_generator.params.register_hook(lambda x:print('gaussian', x))
        self.kernel_size = kernel_size
        self.pad = (kernel_size // 2)
        
    def forward(self, x, perspective):
        sigma_map = self.sigma_map_generator(perspective).clamp(min=1e-4)
        x = torch.nn.functional.pad(x, [self.pad, self.pad, self.pad, self.pad ])
        return GaussianSmoothFunction.apply(x, sigma_map, self.kernel_size, self.kernel_size)

