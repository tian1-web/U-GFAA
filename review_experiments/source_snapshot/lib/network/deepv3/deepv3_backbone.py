import logging
import torch
from torch import nn

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

        # AMP 安全：强制 float32 做 FFT
        x_float = x.float()

        fft_feat = torch.fft.rfft2(x_float, norm='ortho')
        mag = torch.abs(fft_feat)
        phase = torch.angle(fft_feat)

        mag_pool = torch.mean(mag, dim=1, keepdim=True)
        phase_pool = torch.mean(phase, dim=1, keepdim=True)
        fft_combined = torch.cat([mag_pool, phase_pool], dim=1)

        spatial_attn = self.spatial_conv(fft_combined.to(x.dtype)).float()
        enhanced_fft = (mag * spatial_attn) * torch.exp(1j * phase)

        freq_feat = torch.fft.irfft2(enhanced_fft, s=original_size, norm='ortho')
        freq_feat = freq_feat.to(x.dtype)

        fusion_weight = self.channel_attention(identity)
        output = identity + fusion_weight * (freq_feat - identity)
        return output


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super().__init__()

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise ValueError(f'output stride of {output_stride} not supported')

        self.features = []
        self.features.append(
            nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            )
        )

        for r in rates:
            self.features.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_dim, reduction_dim, kernel_size=3,
                        dilation=r, padding=r, bias=False
                    ),
                    Norm2d(reduction_dim),
                    nn.ReLU(inplace=True)
                )
            )

        self.features = nn.ModuleList(self.features)

        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim),
            nn.ReLU(inplace=True)
        )

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


def normalize_backbone_name(name: str) -> str:
    n = name.lower().replace('_', '-')
    if n in ['wideresnet38', 'wide-resnet38', 'wide-resnet-38', 'wrn38']:
        return 'wideresnet38'
    if n in ['resnet50', 'resnet-50']:
        return 'resnet-50'
    if n in ['resnet101', 'resnet-101']:
        return 'resnet-101'
    raise ValueError(f'Unsupported backbone: {name}')


class DeepOODV3Plus(nn.Module):
    """
    OoD-capable DeepLabV3+ with switchable backbones:
      - WideResNet38
      - ResNet50
      - ResNet101

    Unified outputs:
      anomaly_score, logit, freq_enhanced_features
    """

    def __init__(self, num_classes, criterion=None, trunk='WideResNet38', variant='D'):
        super().__init__()
        self.criterion = criterion
        self.variant = variant
        self.trunk = normalize_backbone_name(trunk)

        logging.debug("DeepOODV3Plus trunk: %s", self.trunk)

        if self.trunk == 'wideresnet38':
            self._build_wrn38()
            feat_ch = 4096
            skip_ch = 128
            self.skip_source = 'wrn_m2'

        elif self.trunk == 'resnet-50':
            self._build_resnet(backbone='resnet-50')
            feat_ch = 2048
            skip_ch = 256
            self.skip_source = 'res_layer1'

        elif self.trunk == 'resnet-101':
            self._build_resnet(backbone='resnet-101')
            feat_ch = 2048
            skip_ch = 256
            self.skip_source = 'res_layer1'

        else:
            raise ValueError(f'Unsupported backbone: {trunk}')

        self.freq_enhancement = FrequencyEnhancement(in_channels=feat_ch)

        self.aspp = _AtrousSpatialPyramidPoolingModule(
            feat_ch, 256, output_stride=8
        )
        self.bot_fine = nn.Conv2d(skip_ch, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)

        self.final = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
        )

        self.ood_head = nn.Conv2d(256, num_classes, kernel_size=1, bias=False)

        initialize_weights(self.aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.final)
        initialize_weights(self.ood_head)

    def _build_wrn38(self):
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

    def _build_resnet(self, backbone='resnet-50'):
        if backbone == 'resnet-50':
            resnet = Resnet.resnet50()
        elif backbone == 'resnet-101':
            resnet = Resnet.resnet101()
        else:
            raise ValueError(f'Unsupported ResNet backbone: {backbone}')

        resnet.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )

        self.layer0 = resnet.layer0
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

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

    def energy_func(self, logit):
        anomaly_score = -(1.0 * torch.logsumexp(logit, dim=1))
        return anomaly_score

    def uncertainty_func_init(self):
        if self.final[-1].weight.shape == self.ood_head.weight.shape:
            self.ood_head.weight.data = self.final[-1].weight.data.clone()

    def _forward_backbone(self, inp):
        if self.trunk == 'wideresnet38':
            x = self.mod1(inp)
            skip = self.mod2(self.pool2(x))   # m2
            x = self.mod3(self.pool3(skip))
            x = self.mod4(x)
            x = self.mod5(x)
            x = self.mod6(x)
            x = self.mod7(x)
            return x, skip

        elif self.trunk in ['resnet-50', 'resnet-101']:
            x0 = self.layer0(inp)
            skip = self.layer1(x0)
            x = self.layer2(skip)
            x = self.layer3(x)
            x = self.layer4(x)
            return x, skip

        else:
            raise RuntimeError(f'Unknown trunk: {self.trunk}')

    def forward(self, inp):
        x_size = inp.size()

        backbone_feat, skip_feat = self._forward_backbone(inp)

        freq_enhanced_features = self.freq_enhancement(backbone_feat)

        dec = self.aspp(freq_enhanced_features)
        dec0_up = self.bot_aspp(dec)
        dec0_fine = self.bot_fine(skip_feat)

        dec0_up = Upsample(dec0_up, skip_feat.size()[2:])
        dec0 = torch.cat([dec0_fine, dec0_up], 1)

        feature = self.final[:-1](dec0)

        dec1 = self.final[-1](feature)
        logit = Upsample(dec1, x_size[2:])

        dec2 = self.ood_head(feature)
        anomaly_score = Upsample(
            self.energy_func(dec2).unsqueeze(1), x_size[2:]
        ).squeeze(1)

        return anomaly_score, logit, freq_enhanced_features