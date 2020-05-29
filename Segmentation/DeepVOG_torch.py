import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import os

class Conv2dTf(nn.Conv2d):
    '''
    Conv2d with the padding behavior from TF
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias, padding_mode)
        self.padding = padding
    
    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
        )
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == 'valid':
            return F.conv2d(input, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)
        elif self.padding == 'same':
            rows_odd, padding_rows = self._compute_padding(input, dim=0)
            cols_odd, padding_cols = self._compute_padding(input, dim=1)
            if rows_odd or cols_odd:
                input = F.pad(input, [0, cols_odd, 0, rows_odd])
            return F.conv2d(input, self.weight, self.bias, self.stride, (padding_rows//2, padding_cols//2), self.dilation, self.groups)

class DeepVOGEncodingBlock(nn.Module):
    '''
    One encoding block in the DeepVOG model
    '''
    
    def __init__(self, in_channels, out_channels, kernel_size, down_sampling=True, skip=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels*2 if down_sampling else out_channels
        self.kernel_size = kernel_size if type(kernel_size)==tuple else (kernel_size, kernel_size)
        self.down_sampling = down_sampling
        self.skip = skip
        self.add_module(f'conv_main', Conv2dTf(in_channels, out_channels, kernel_size))
        self.add_module(f'bn_main', nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1))
        self.add_module(f'act_main', nn.ReLU())
        if down_sampling:
            self.add_module(f'conv_down', nn.Conv2d(out_channels, out_channels*2, 2, stride=2))
            self.add_module(f'bn_down', nn.BatchNorm2d(out_channels*2, eps=0.001, momentum=0.1))
            self.add_module(f'act_down', nn.ReLU())
        
    def forward(self, x):
        x = self._modules['conv_main'](x)
        x = self._modules['bn_main'](x)
        x = self._modules['act_main'](x)
        if self.down_sampling:
            x_down = self._modules['conv_down'](x)
            x_down = self._modules['bn_down'](x_down)
            x_down = self._modules['act_down'](x_down)
        else:
            x_down = x
        if self.skip:
            return x_down, x
        else:
            return x_down

class DeepVOGDecodingBlock(nn.Module):
    '''
    One decoding block in the DeepVOG model
    '''
    
    def __init__(self, in_channels, out_channels, kernel_size, jump_channels=0, up_sampling=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if type(kernel_size)==tuple else (kernel_size, kernel_size)
        self.jump_channels = jump_channels
        self.up_sampling = up_sampling
        self.add_module(f'conv_main', Conv2dTf(in_channels+jump_channels, out_channels, kernel_size))
        self.add_module(f'bn_main', nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1))
        self.add_module(f'act_main', nn.ReLU())
        if up_sampling:
            self.add_module(f'conv_up', nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2))
            self.add_module(f'bn_up', nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1))
            self.add_module(f'act_up', nn.ReLU())
        
    def forward(self, x, x_jump=None):
        if self.jump_channels>0:
            if x_jump is None:
                raise ValueError('Jump connected defined in model but not provided.')
            if x_jump.shape[1]!=self.jump_channels:
                raise ValueError(f'Expected {self.jump_channels} channels in the jump connection, got {x_jump.shape[1]}')
            x = torch.cat((x, x_jump), 1)
        x = self._modules['conv_main'](x)
        x = self._modules['bn_main'](x)
        x = self._modules['act_main'](x)
        if self.up_sampling:
            x_up = self._modules['conv_up'](x)
            x_up = self._modules['bn_up'](x_up)
            x_up = self._modules['act_up'](x_up)
        else:
            x_up = x
        return x_up

class DeepVOG(nn.Module):
    '''
    DeepVOG model. Output has three channels. 
    The first two channels represent two one-hot vectors (pupil and non-pupil)
    The third layer contains all zeros in all cases
    '''
    
    def __init__(self, in_channels=3, out_channels=3, down_channels=(16, 32, 64, 128), up_channels=(256, 256, 128, 64, 32), kernel_size=(10, 10)):
        super().__init__()
        self.layers_down = len(down_channels)
        self.layers_up = len(up_channels)
        if self.layers_up!=self.layers_down+1:
            raise ValueError('Number of upsampling layers must be greater than the number of downsampling layer by 1.')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_channels = down_channels
        self.up_channels = up_channels
        self.down_in_channels = (in_channels, *tuple([s*2 for s in down_channels[:-1]]))
        self.down_out_channels = tuple([s*2 for s in down_channels])
        self.up_in_channels = (self.down_out_channels[-1], *up_channels[:-1])
        self.up_out_channels = up_channels
        self.jump_channels = (0, *down_channels[::-1])
        self.up_sampling = tuple([k<self.layers_up-1 for k in range(self.layers_up)])
        for k in range(self.layers_down):
            self.add_module(f'down_{k+1}', DeepVOGEncodingBlock(self.down_in_channels[k], self.down_out_channels[k]//2, kernel_size))
        for k in range(self.layers_up):
            self.add_module(f'up_{k+1}', DeepVOGDecodingBlock(self.up_in_channels[k], self.up_out_channels[k], kernel_size, self.jump_channels[k], self.up_sampling[k]))
        self.add_module(f'conv_out', nn.Conv2d(up_channels[-1], out_channels, (1, 1)))
        self.add_module(f'act_out', nn.Softmax2d())
        
    def forward(self, x):
        if x.shape[1]!=self.in_channels:
            raise ValueError(f'Expected {self.in_channels} channels in the input, got {x.shape[1]}')
        x_skips = []
        for k in range(self.layers_down):
            x, x_skip = self._modules[f'down_{k+1}'](x)
            #print(f'down_{k+1}: {x.shape}, skip_{k+1}: {x_skip.shape}')
            x_skips.append(x_skip)
        x_skips.append(None)
        x_skips = x_skips[::-1]
        for k in range(self.layers_up):
            x = self._modules[f'up_{k+1}'](x, x_skips[k])
            #print(f'up_{k+1}: {x.shape}')
        x = self._modules[f'conv_out'](x)
        x = self._modules[f'act_out'](x)
        #print(f'out: {x.shape}')
        return x
