import torch
import numpy as np
import torchvision.transforms.functional as TF

def amp_spectrum_mix(img_source, img_style, alpha=None, alpha_range=(0.0, 0.5)):
    """
    [V1 版本] 基于全图的幅度谱混合 (Global Amplitude Mixup).
    """
    # 1. FFT 变换
    fft_src = torch.fft.rfft2(img_source.clone(), norm='backward')
    fft_style = torch.fft.rfft2(img_style.clone(), norm='backward')

    # 2. 分解
    amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
    amp_style, _ = torch.abs(fft_style), torch.angle(fft_style)

    # 3. 确定混合比例
    if alpha is None:
        alpha = np.random.uniform(alpha_range[0], alpha_range[1])
    
    # 4. 幅度混合
    amp_mix = (1 - alpha) * amp_src + alpha * amp_style

    # 5. 结构重组 (相位锁定)
    fft_mix = amp_mix * torch.exp(1j * pha_src)

    # 6. iFFT
    img_aug = torch.fft.irfft2(fft_mix, s=img_source.shape[-2:], norm='backward')

    return img_aug


def uncertainty_aware_amp_mix(img_source, img_style, uncertainty_map=None, strength=0.5):
    """
    [V3 最终版] 不确定性引导的平滑局部频域增强 (Smoothed U-SAFA).
    
    改进点：
    1. 引入高斯模糊平滑 Mask，消除混合边界的伪影，保护小物体边缘。
    2. 支持 uncertainty_map=None，此时执行随机强度的全图混合。
    """
    B, C, H, W = img_source.shape
    
    # 1. FFT 变换 (转换到频域)
    fft_src = torch.fft.rfft2(img_source.float(), norm='backward')
    fft_style = torch.fft.rfft2(img_style.float(), norm='backward')
    
    # 2. 分解特征
    # 保留原图相位 (结构)，提取风格图幅度 (风格)
    amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
    amp_style = torch.abs(fft_style)
    
    # 3. 生成混合掩码 (Mask Generation)
    if uncertainty_map is not None:
        # === 模式 A: 局部不确定性引导混合 ===
        # 3.1 插值调整大小以匹配频域尺寸 (rfft2 的宽度是 W/2 + 1)
        with torch.no_grad():
            # 插值
            mask = torch.nn.functional.interpolate(uncertainty_map, size=amp_src.shape[-2:], mode='bilinear')
            
            # [关键改进] 3.2 高斯模糊平滑
            # kernel_size 必须是奇数，sigma 控制模糊程度
            mask = TF.gaussian_blur(mask, kernel_size=7, sigma=2.0)
            
            # 3.3 应用强度系数
            mask = mask * strength
    else:
        # === 模式 B: 全图随机混合 (Global Mix) ===
        # 随机采样一个全局强度，例如 0.1 ~ 0.5
        current_strength = np.random.uniform(0.1, 0.5)
        mask = torch.ones_like(amp_src) * current_strength
        mask = mask.to(img_source.device)

    # 4. 执行混合 (Soft Mixing)
    amp_mix = (1 - mask) * amp_src + mask * amp_style
    
    # 5. 结构重组 (Reconstruction) - 锁定相位
    fft_mix = amp_mix * torch.exp(1j * pha_src)
    
    # 6. iFFT 逆变换
    img_aug = torch.fft.irfft2(fft_mix, s=(H, W), norm='backward')
    img_aug = img_aug.to(img_source.dtype)
    return img_aug