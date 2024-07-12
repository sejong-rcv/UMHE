# --------------------------------------------------------
# High Resolution Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Rao Fu, RainbowSecret
# --------------------------------------------------------

import os
import pdb
import logging
import torch.nn as nn

class Final_layer(nn.Module):

    def __init__(self, dim=32):
        super().__init__()
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.norm = LayerNorm(dim, eps=1e-6)

        self.norm = nn.LayerNorm(dim, eps=1e-6)

        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim, bias= False)

        self.proj = nn.Linear(dim, dim//4, bias= False)

    def forward(self, x):

        x = self.norm(x)

        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)

        attn = out.softmax(dim=-1)
        
        res = x * attn
        x = x + res

        return self.proj(x)

class BasicBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, layer_scale_init_value=1e-6):
        super().__init__()

        self.dwconv = nn.Conv2d(inplanes, planes, kernel_size=7, padding=3, groups=inplanes) # depthwise conv
        # self.norm = LayerNorm(dim, eps=1e-6)
        self.norm = nn.BatchNorm2d(planes)
        self.pwconv1 = nn.Conv2d(planes, 4 * planes, 1) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * planes, planes, 1)
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
        #                             requires_grad=True) if layer_scale_init_value > 0 else None
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        x = self.dwconv(x)
        # x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # if self.gamma is not None:
        #     x = self.gamma * x
        # x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input# + self.drop_path(x)

        return x

# BN_MOMENTUM = 0.1


# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(
#         in_planes,
#         out_planes,
#         kernel_size=3,
#         stride=stride,
#         padding=dilation,
#         groups=groups,
#         bias=False,
#         dilation=dilation,
#     )


# class BasicBlock(nn.Module):
#     """Only replce the second 3x3 Conv with the TransformerBlocker"""

#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)

#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out
