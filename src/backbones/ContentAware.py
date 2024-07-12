import torch
import torch.nn as nn
import torchvision.models as models
import kornia

from torchvision.transforms.functional import to_pil_image
from src.utils.phase_congruency import _phase_congruency
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import math
from .ResNet34 import resnet34
import torch.nn.functional as F

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()
        self.pwconv = nn.Conv2d(in_dim, out_dim, 1, 1, bias=True)

        self.weight = nn.Parameter(torch.randn(out_dim, out_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(out_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):

        if self.activation:
            if input.dim() > 3:
                out = F.conv2d(input, self.weight[:, :, None, None] * self.scale)
            else:
                out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            if input.dim() > 3:
                out = F.conv2d(input, self.weight[:, :, None, None] * self.scale,
                               bias=self.bias * self.lr_mul
                )
            else:
                out = F.linear(
                    input, self.weight * self.scale, bias=self.bias * self.lr_mul
                )

        return out

class Attnblock(nn.Module):

    def __init__(self, dim=16, fix_mask=False, normalization_strength=-1):
        super(Attnblock, self).__init__()
        self.fix_mask = fix_mask
        self.normalization_strength = normalization_strength

        self.pwconv = nn.Conv2d(dim, dim, 1)
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=3, groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    @staticmethod
    def __normalize_mask(mask, strength=0.5):
        batch_size, c_m, c_h, c_w = mask.size()
        max_value = mask.reshape(batch_size, -1).max(1)[0]
        max_value = max_value.reshape(batch_size, 1, 1, 1)
        mask = mask / (max_value * strength)
        mask = torch.clamp(mask, 0, 1)
        return mask

    def forward(self, x):
        if self.fix_mask:
            out = torch.ones_like(x)
        else:

            x = self.pwconv(x)

            x_1 = self.conv0(x)
            x_2 = self.conv_spatial(x)

            out = x_1 + x_2

            out = self.conv1(out)

            assert out.shape[-2:] == x.shape[-2:], 'Mask and input image should have the same w/h'

            # Normalize mask
            if self.normalization_strength > 0:
                out = self.__normalize_mask(out, strength=self.normalization_strength)

        return out

class MaskPredictor(nn.Module):

    def __init__(self, dim=16, fix_mask=False, normalization_strength=-1):
        super(MaskPredictor, self).__init__()

        self.pwconv = nn.Conv2d(dim, dim, 1)

    def make_interval(self, start=0.01, end=1, depth=32):
        
        return torch.linspace(start, end, 32, requires_grad=False).unsqueeze(0).unsqueeze(2).unsqueeze(3)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.pwconv(x)
        grid = self.make_interval().repeat(b, 1, h, w).float().to(x.device)

        z = F.softmax(x, dim=1)
        z = z * grid
        z = torch.sum(z, dim=1, keepdim=True)

        return z

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim=16, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv

        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        x = self.dwconv(x)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        x = input + self.drop_path(x)
        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=440, patch_size=4, stride=4, in_chans=1, embed_dim=16):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = nn.BatchNorm2d(embed_dim)
        self.linear_embed = nn.Conv2d(embed_dim, embed_dim*2, 1, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        return x, H, W

class Extractor(nn.Module):
    def __init__(self, img_size=440, in_chans=1, num_classes=1000, embed_dim=16, drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.BatchNorm2d, depths=2, num_stages=4, flag=False):
        super().__init__()
        if flag == False:
            self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        patch_embed = OverlapPatchEmbed(img_size=img_size, patch_size=4, stride=4,
                                        in_chans=in_chans, embed_dim=embed_dim)

        block = nn.ModuleList([Block(dim=embed_dim) for i in range(depths)])        

        mask_extractor = nn.Conv2d(embed_dim, 1, 1)
        
        self.proj = nn.Conv2d(embed_dim, 1, 1)
        self.spatial_attn = Attnblock(embed_dim)
        self.activation = nn.Sigmoid()

        self.tanh = nn.Tanh()

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)
        setattr(self, f"mask_extractor", mask_extractor)

    def forward(self, x):
        out, _, _ = self.patch_embed(x)

        for block in self.block:
            out = block(out)

        spatial_attn = self.spatial_attn(out)

        mask = self.activation(self.mask_extractor(spatial_attn))

        feature = self.proj(out)
        b,c,h,w = feature.shape

        return feature, mask#, content

    def calc_mean_std(self, feat, eps=1e-5):

        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()

        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

        return feat_mean, feat_std


    def adaptive_instance_normalization(self, source_feat, target_feat):

        size = source_feat.size()
        b, c, h, w = source_feat.shape

        source_mean, source_std = self.calc_mean_std(source_feat)
        target_mean, target_std = self.calc_mean_std(target_feat)
        
        bias, scale = self.calc_mean_std(torch.cat([source_feat.view(b,c,-1), target_feat.view(b,c,-1)], dim=-1))

        normalized_source_feat = (source_feat - source_mean.expand(
            size)) / source_std.expand(size)
        normalized_target_feat = (target_feat - target_mean.expand(
            size)) / target_std.expand(size)
        
        
        source = normalized_source_feat * scale.expand(size) + bias.expand(size)
        target = normalized_target_feat * scale.expand(size) + bias.expand(size)
        
        return source, target

class Model(nn.Module):

    def __init__(self, isresnet=True, **kwargs):
        super(Model, self).__init__()

        self.patch_keys = kwargs['PATCH_KEYS']
        self.mask_keys = kwargs['MASK_KEYS']
        self.feature_keys = kwargs['FEATURE_KEYS']
        self.target_keys = kwargs['TARGET_KEYS']

        mask_norm_strength = kwargs['MASK_NORMALIZATION_STRENGTH'] if 'MASK_NORMALIZATION_STRENGTH' in kwargs else -1

        self.feature_extractor = Extractor(img_size=448)

        self.variant = str.lower(kwargs['VARIANT'])
        assert 'oneline' in self.variant or 'doubleline' in self.variant, 'Only OneLine or DoubleLine variant is' \
                                                                          'supported'

        # Init weights if we're using pretrained resnet
        pretrained_resnet = kwargs['PRETRAINED_RESNET']
        if pretrained_resnet:
            self.init()

        # Get ResNet model without first and last elem
        if isresnet :
            print("Extractor : resnet34")
            self.model = resnet34()

        if not pretrained_resnet:
            self.init()

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def print_params(self, net):
        if isinstance(net, list):
            net = net[0]
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            # print(net)
            print('Total number of parameters: %d' % num_params)

    def _forward(self, input_1, input_2):

        inputs = torch.cat([input_1, input_2], dim=0)
        f, m = self.feature_extractor(inputs)

        f1, f2 = f.chunk(chunks=2, dim=0)

        b, c, h, w = f.shape

        m1, m2 = m.chunk(chunks=2, dim=0)

        g = torch.cat(torch.mul(f, m).chunk(chunks=2, dim=0), dim=1)
        g1, g2 = g.chunk(chunks=2, dim=1)

        o = self.model(g)

        if isinstance(o, list):
            for i in range(len(o)):
                o[i] = o[i].reshape(-1, 4, 2)
        else:
            o = o.reshape(-1, 4, 2)

        return m1, f1, m2, f2, g1, g2, o, f

    def forward(self, data):
        
        e1, e2 = self.patch_keys
        m1, m2 = self.mask_keys
        f1, f2 = self.feature_keys
        o1 = self.target_keys[0]

        # Main pass
        data[m1], data[f1], data[m2], data[f2], g1, g2, data[o1], c = self._forward(data[e1], data[e2])

        # Auxiliary pass
        if self.variant == 'doubleline':
            g = torch.cat([g2, g1], axis=1)

            o2 = self.target_keys[1]
            o = self.model(g)
            if isinstance(o, list):
                for i in range(len(o)):
                    o[i] = o[i].reshape(-1, 4, 2)
            else:
                o = o.reshape(-1, 4, 2)
            data[o2] = o

        # Return
        return data

    def predict_homography(self, data):
        
        # Get keys
        e1, e2 = self.patch_keys
        o1 = self.target_keys[0]
        m1, m2 = self.mask_keys

        # Main pass
        data[m1], _, data[m2], _, g1, g2, data[o1], _ = self._forward(data[e1], data[e2])
        
        return data

    def retrieve_weights(self):
        weights = {}
        for name, parameter in self.resnet34.named_parameters():
            weights[name] = parameter.data
        return weights
