import torch
import numpy as np

def amp_spectrum_mix(img_source, img_style, alpha=None, alpha_range=(0.0, 0.5)):
    """
    基于可靠性约束的幅度谱混合 (Reliability-Aware Amplitude Spectrum Mixup).
    
    核心逻辑：
    1. 保留 img_source 的相位谱 (Phase Spectrum) -> 确保几何结构、边缘与 Mask 100% 对齐。
    2. 混合 img_source 和 img_style 的幅度谱 (Amplitude Spectrum) -> 引入异域风格。
    3. 限制混合比例 alpha -> 避免图像过度失真，保留“自然性”。

    Args:
        img_source (torch.Tensor): 原始图像张量, shape (C, H, W).
        img_style (torch.Tensor): 风格参考图像张量 (如 ControlNet 生成图), shape (C, H, W).
        alpha (float, optional): 手动指定的混合比例. 如果为 None，则从 alpha_range 中随机采样.
        alpha_range (tuple): 混合比例的范围 (min, max). 建议上限不要超过 0.5.

    Returns:
        img_aug (torch.Tensor): 增强后的图像张量, shape (C, H, W).
    """
    
    # 1. FFT 变换 (转到频域)
    # 使用 rfft2 处理实数图像，效率比 fft2 更高，输出只包含一半频率(共轭对称)
    # norm='backward' 是 PyTorch 默认的归一化方式
    fft_src = torch.fft.rfft2(img_source.clone(), norm='backward')
    fft_style = torch.fft.rfft2(img_style.clone(), norm='backward')

    # 2. 分解 幅度(Amp) 和 相位(Pha)
    # abs() 获取复数模长(幅度)，angle() 获取复数辐角(相位)
    amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
    amp_style, _ = torch.abs(fft_style), torch.angle(fft_style) # 不需要 style 的相位

    # 3. 确定混合比例 alpha (The Reliability Constraint)
    # 如果没有指定 alpha，则从均匀分布 U(min, max) 中采样
    # 这里的上限建议设为 0.5，保证原图的信息至少保留 50%，起到“保底”作用
    if alpha is None:
        alpha = np.random.uniform(alpha_range[0], alpha_range[1])
    
    # 4. 混合幅度谱 (Style Mixing)
    # 公式：Amp_mix = (1 - alpha) * Amp_src + alpha * Amp_style
    # 这样既保留了原图的纹理基底，又融入了 Style 图(如雪天)的频域特征
    amp_mix = (1 - alpha) * amp_src + alpha * amp_style

    # 5. 结构重组 (Reconstruction)
    # 关键点：必须使用 pha_src (原图相位) 来组合！
    # 这样逆变换回去的图像，其物体边缘、位置绝对不会发生任何位移。
    # Euler公式: F = Amp * e^(j * Phase)
    fft_mix = amp_mix * torch.exp(1j * pha_src)

    # 6. iFFT 逆变换 (回到像素域)
    # s 参数指定输出尺寸，确保和输入一致
    img_aug = torch.fft.irfft2(fft_mix, s=img_source.shape[-2:], norm='backward')

    return img_aug