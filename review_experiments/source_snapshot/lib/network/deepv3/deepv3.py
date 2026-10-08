"""
# Code Adapted from:
# https://github.com/sthalles/deeplab_v3
#
# MIT License
#
# Copyright (c) 2018 Thalles Santos Silva
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
"""
import logging
import torch
from torch import nn
from . import SEresnext
from . import Resnet
from .wider_resnet import wider_resnet38_a2
from .mynn import initialize_weights, Norm2d, Upsample

class FrequencyEnhancement(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        original_size = x.shape[-2:]
    
        # [关键修改] 强制转换为 float32，避免 AMP 下 FP16 FFT 的尺寸限制
        x_float = x.float()

        fft_feat = torch.fft.rfft2(x_float, norm='ortho')
        mag = torch.abs(fft_feat)
        phase = torch.angle(fft_feat)

        mag_pool = torch.mean(mag, dim=1, keepdim=True)
        phase_pool = torch.mean(phase, dim=1, keepdim=True)
        fft_combined = torch.cat([mag_pool, phase_pool], dim=1)

        # 空间注意力部分建议保持在 float32 下计算，或者是 x.dtype
        # 这里为了稳妥，将输入转回原始 dtype 进行卷积，输出再转回 float32
        spatial_attn = self.spatial_conv(fft_combined.to(x.dtype))
        spatial_attn = spatial_attn.float() 
    
        # 增强操作 (FP32)
        enhanced_fft = (mag * spatial_attn) * torch.exp(1j * phase)
        
        # 逆变换 (FP32)
        freq_feat = torch.fft.irfft2(enhanced_fft, s=original_size, norm='ortho')

        # [关键修改] 转回原始精度 (如 FP16) 以匹配后续网络
        freq_feat = freq_feat.to(x.dtype)

        fusion_weight = self.channel_attention(identity)
        output = identity + fusion_weight * (freq_feat - identity)
        return output

class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class DeepV3Plus(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='seresnext-50', criterion=None, variant='D',
                 skip='m1', skip_num=48):
        super(DeepV3Plus, self).__init__()
        self.criterion = criterion
        self.variant = variant
        self.skip = skip
        self.skip_num = skip_num

        if trunk == 'seresnext-50':
            resnet = SEresnext.se_resnext50_32x4d()
        elif trunk == 'seresnext-101':
            resnet = SEresnext.se_resnext101_32x4d()
        elif trunk == 'resnet-50':
            resnet = Resnet.resnet50()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-101':
            resnet = Resnet.resnet101()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError("Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if self.variant == 'D':
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        elif self.variant == 'D16':
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            print("Not using Dilation ")

        self.aspp = _AtrousSpatialPyramidPoolingModule(2048, 256,
                                                       output_stride=8)

        if self.skip == 'm1':
            self.bot_fine = nn.Conv2d(256, self.skip_num, kernel_size=1, bias=False)
        elif self.skip == 'm2':
            self.bot_fine = nn.Conv2d(512, self.skip_num, kernel_size=1, bias=False)
        else:
            raise Exception('Not a valid skip')

        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)

        self.final = nn.Sequential(
            nn.Conv2d(256 + self.skip_num, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final)

    def forward(self, x, gts=None):

        x_size = x.size()
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        xp = self.aspp(x4)

        dec0_up = self.bot_aspp(xp)
        if self.skip == 'm1':
            dec0_fine = self.bot_fine(x1)
            dec0_up = Upsample(dec0_up, x1.size()[2:])
        else:
            dec0_fine = self.bot_fine(x2)
            dec0_up = Upsample(dec0_up, x2.size()[2:])

        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        dec1 = self.final(dec0)
        main_out = Upsample(dec1, x_size[2:])

        if self.training:
            return self.criterion(main_out, gts)

        return main_out

class DeepWV3Plus(nn.Module):
    """
    Wide_resnet version of DeepLabV3
    mod1
    pool2
    mod2 str2
    pool3
    mod3-7

      structure: [3, 3, 6, 3, 1, 1]
      channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048),
                  (1024, 2048, 4096)]
    """

    def __init__(self, num_classes, criterion = None, trunk='WideResnet38'):
        super(DeepWV3Plus, self).__init__()
        logging.debug("Trunk: %s", trunk)
        wide_resnet = wider_resnet38_a2(classes=1000, dilation=True)
        self.mod1 = wide_resnet.mod1
        self.mod2 = wide_resnet.mod2
        self.mod3 = wide_resnet.mod3
        self.mod4 = wide_resnet.mod4
        self.mod5 = wide_resnet.mod5
        self.mod6 = wide_resnet.mod6
        self.mod7 = wide_resnet.mod7
        self.pool2 = wide_resnet.pool2
        self.pool3 = wide_resnet.pool3
        del wide_resnet

        # 频域增强模块实例化
        self.freq_enhancement = FrequencyEnhancement(in_channels=4096)

        self.aspp = _AtrousSpatialPyramidPoolingModule(4096, 256,
                                                       output_stride=8)
        self.bot_fine = nn.Conv2d(128, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)

        self.final = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        self.ood_head = nn.Conv2d(256, num_classes, kernel_size=1, bias=False)

        self.criterion = criterion

    def energy_func(self, logit):
        anomaly_score = -(1. * torch.logsumexp(logit, dim=1))
        return anomaly_score

    def uncertainty_func_init(self,):
        self.ood_head.weight.data = self.final[-1].weight.data.clone()

    def forward(self, inp):

        x_size = inp.size()
        x = self.mod1(inp)
        m2 = self.mod2(self.pool2(x))
        x = self.mod3(self.pool3(m2))
        x = self.mod4(x)
        x = self.mod5(x)
        x = self.mod6(x)
        x = self.mod7(x)

        # 频域增强模块调用
        freq_enhanced_features = self.freq_enhancement(x)

        # 修改 ASPP 输入为原始特征 x
        dec = self.aspp(x) 
        dec0_up = self.bot_aspp(dec)
        dec0_fine = self.bot_fine(m2)

        dec0_up = Upsample(dec0_up, m2.size()[2:])
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)

        feature = self.final[:-1](dec0)

        dec1 = self.final[-1](feature)
        logit = Upsample(dec1, x_size[2:])

        dec2 = self.ood_head(feature)
        anomaly_score = Upsample(self.energy_func(dec2).unsqueeze(1), x_size[2:]).squeeze(1)

        # 返回包含频域增强特征的完整输出
        return anomaly_score, logit, freq_enhanced_features


def DeepSRNX50V3PlusD_m1(num_classes, criterion):
    """
    SEResnet 50 Based Network
    """
    return DeepV3Plus(num_classes, trunk='seresnext-50', criterion=criterion, variant='D',
                      skip='m1')

def DeepR50V3PlusD_m1(num_classes, criterion):
    """
    Resnet 50 Based Network
    """
    return DeepV3Plus(num_classes, trunk='resnet-50', criterion=criterion, variant='D', skip='m1')


def DeepSRNX101V3PlusD_m1(num_classes, criterion):
    """
    SeResnext 101 Based Network
    """
    return DeepV3Plus(num_classes, trunk='seresnext-101', criterion=criterion, variant='D',
                      skip='m1')

def DeepR101V3PlusD_OS8(num_classes, criterion):
    """
    Resnet 101 Based Network
    """
    return DeepV3Plus(num_classes, trunk='resnet-101', criterion=criterion,
                    variant='D', skip='m1')