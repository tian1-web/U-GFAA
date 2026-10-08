import os
import sys
import argparse
from typing import Tuple

# =========================================================
# 先解析我们自己的参数，再把安全参数传给仓库全局 parser
# =========================================================
pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument('--cfg', type=str, default='exps/DeepLab.yaml')
pre_parser.add_argument('--id', type=str, default='vis_clean_ood')
pre_parser.add_argument('--dataset', type=str, required=True,
                        choices=['RoadAnomaly', 'RoadObstacle21'])
pre_parser.add_argument('--weight_path', type=str, required=True)
pre_parser.add_argument('--output_dir', type=str, required=True)
pre_parser.add_argument('--device', type=str, default='cuda')
pre_parser.add_argument('--use_dp', action='store_true')
pre_parser.add_argument('--num_workers', type=int, default=0)
pre_parser.add_argument('--max_samples', type=int, default=12)
pre_parser.add_argument('--start_index', type=int, default=0)

# 可视化参数：已经调成更接近你参考图的默认值
pre_parser.add_argument('--low_percentile', type=float, default=5.0)
pre_parser.add_argument('--high_percentile', type=float, default=99.5)
pre_parser.add_argument('--gamma', type=float, default=2.2)
pre_parser.add_argument('--suppress_threshold', type=float, default=0.35,
                        help='低于该值的响应直接压零，背景更干净')
pre_parser.add_argument('--smooth_kernel', type=int, default=5)
pre_parser.add_argument('--max_alpha', type=float, default=0.90)
pre_parser.add_argument('--cmap', type=str, default='jet')
pre_parser.add_argument('--show_title', action='store_true')
pre_parser.add_argument('--invert_score', action='store_true',
                        help='如果发现异常区域反而不亮，可以加这个参数反转分数')

args, _unknown = pre_parser.parse_known_args()

# 让仓库自己的 parse_arg.py 只看到这些参数
sys.argv = [
    sys.argv[0],
    '--cfg', args.cfg,
    '--id', args.id,
    '--test_dataset', args.dataset,
    '--weight_path', args.weight_path,
]

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from lib.network.deepv3 import DeepWV3Plus
from lib.dataset.cityscapes import DiverseCityscapes
from lib.dataset.anomaly import RoadAnomaly, RoadObstacle21


# =========================================================
# 测试变换
# =========================================================
class TestTransform:
    def __init__(self):
        self.normalize = T.Normalize(
            mean=DiverseCityscapes.mean,
            std=DiverseCityscapes.std
        )

    def __call__(self, image: Image.Image, target: Image.Image):
        image = T.ToTensor()(image)
        image = self.normalize(image)

        target_np = np.array(target, dtype=np.uint8)
        target = torch.from_numpy(target_np).long()
        return image, target


# =========================================================
# 只取一段子集，方便可视化
# =========================================================
class SliceDataset(Dataset):
    def __init__(self, base_dataset, start_index=0, max_samples=12):
        self.base_dataset = base_dataset
        self.indices = list(range(start_index, min(len(base_dataset), start_index + max_samples)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]


# =========================================================
# 构建数据集
# =========================================================
def build_dataset(name: str):
    tf = TestTransform()

    if name == 'RoadAnomaly':
        ds = RoadAnomaly(transform=tf)
    elif name == 'RoadObstacle21':
        ds = RoadObstacle21(transform=tf)
    else:
        raise ValueError(f'Unsupported dataset: {name}')

    return ds


# =========================================================
# 加载模型
# =========================================================
def load_model(weight_path: str, device: str = 'cuda', use_dp: bool = False):
    model = DeepWV3Plus(19)

    if use_dp and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    checkpoint = torch.load(weight_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if use_dp and torch.cuda.device_count() > 1:
            name = k if k.startswith('module.') else 'module.' + k
        else:
            name = k.replace('module.', '')
        new_state_dict[name] = v

    incompatible = model.load_state_dict(new_state_dict, strict=False)
    print(f'[Info] Loaded checkpoint: {weight_path}')

    if hasattr(incompatible, 'missing_keys') and len(incompatible.missing_keys) > 0:
        print(f'[Warn] Missing keys ({len(incompatible.missing_keys)}): {incompatible.missing_keys[:10]}')
    if hasattr(incompatible, 'unexpected_keys') and len(incompatible.unexpected_keys) > 0:
        print(f'[Warn] Unexpected keys ({len(incompatible.unexpected_keys)}): {incompatible.unexpected_keys[:10]}')

    model = model.to(device)
    model.eval()
    return model


# =========================================================
# 提取异常分数图
# 兼容 U-GFAA forward: anomaly_score, logit, feature
# =========================================================
def extract_anomaly_score(outputs):
    if torch.is_tensor(outputs):
        if outputs.ndim == 3:
            return outputs
        if outputs.ndim == 4 and outputs.shape[1] == 1:
            return outputs.squeeze(1)
        raise RuntimeError(f'Unsupported tensor output shape: {outputs.shape}')

    if isinstance(outputs, (list, tuple)):
        # 优先找 [B,H,W]
        for out in outputs:
            if torch.is_tensor(out) and out.ndim == 3:
                return out
        # 再找 [B,1,H,W]
        for out in outputs:
            if torch.is_tensor(out) and out.ndim == 4 and out.shape[1] == 1:
                return out.squeeze(1)

        shapes = []
        for out in outputs:
            if torch.is_tensor(out):
                shapes.append(tuple(out.shape))
            else:
                shapes.append(type(out).__name__)
        raise RuntimeError(f'Cannot find anomaly score in outputs: {shapes}')

    raise RuntimeError(f'Unsupported output type: {type(outputs)}')


# =========================================================
# 反归一化图像
# =========================================================
def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    mean = np.array(DiverseCityscapes.mean).reshape(3, 1, 1)
    std = np.array(DiverseCityscapes.std).reshape(3, 1, 1)

    img = img_tensor.cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0.0, 1.0)
    img = np.transpose(img, (1, 2, 0))
    return img


# =========================================================
# 简单均值平滑
# =========================================================
def smooth_score_map(score_map: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1:
        return score_map.astype(np.float32)

    score_map = score_map.astype(np.float32)
    pad = k // 2
    padded = np.pad(score_map, ((pad, pad), (pad, pad)), mode='reflect')
    out = np.zeros_like(score_map, dtype=np.float32)

    for i in range(score_map.shape[0]):
        for j in range(score_map.shape[1]):
            out[i, j] = padded[i:i+k, j:j+k].mean()

    return out


# =========================================================
# 更适合论文图的热力图处理
# =========================================================
def prepare_score_map_for_vis(
    score_map: np.ndarray,
    low_percentile: float = 5.0,
    high_percentile: float = 99.5,
    gamma: float = 2.2,
    suppress_threshold: float = 0.35,
    smooth_kernel: int = 5,
    invert_score: bool = False,
) -> np.ndarray:
    score_map = score_map.astype(np.float32)

    if invert_score:
        score_map = -score_map

    low = np.percentile(score_map, low_percentile)
    high = np.percentile(score_map, high_percentile)

    if high - low < 1e-8:
        return np.zeros_like(score_map, dtype=np.float32)

    # 百分位裁剪
    score_map = np.clip(score_map, low, high)
    score_map = (score_map - low) / (high - low + 1e-8)

    # gamma增强，只突出高响应
    score_map = np.power(score_map, gamma)

    # 轻度平滑，让热力图块状感更自然
    score_map = smooth_score_map(score_map, k=smooth_kernel)

    # 背景抑制：低响应直接压零
    score_map = np.where(score_map > suppress_threshold, score_map, 0.0)

    return np.clip(score_map, 0.0, 1.0).astype(np.float32)


# =========================================================
# 保存单张图
# 两列：Input / U-GFAA Heatmap
# =========================================================
def save_visualization(
    image_np: np.ndarray,
    score_vis: np.ndarray,
    save_path: str,
    title: str = '',
    cmap: str = 'jet',
    max_alpha: float = 0.90,
    show_title: bool = False
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # 1. 原图
    axes[0].imshow(image_np)
    axes[0].set_title('Input')
    axes[0].axis('off')

    # 2. 热力图叠加
    axes[1].imshow(image_np)

    # 动态透明度：只有高响应区域才明显着色
    alpha_map = np.clip(score_vis, 0.0, 1.0)
    alpha_map = np.power(alpha_map, 1.2) * max_alpha

    axes[1].imshow(
        score_vis,
        cmap=cmap,
        alpha=alpha_map
    )
    axes[1].set_title('U-GFAA Heatmap')
    axes[1].axis('off')

    if show_title and title:
        fig.suptitle(title, fontsize=13)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


# =========================================================
# 主流程
# =========================================================
def main():
    os.makedirs(args.output_dir, exist_ok=True)

    base_ds = build_dataset(args.dataset)
    vis_ds = SliceDataset(base_ds, start_index=args.start_index, max_samples=args.max_samples)

    loader = DataLoader(
        vis_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f'[Info] Dataset       : {args.dataset}')
    print(f'[Info] Total in base : {len(base_ds)}')
    print(f'[Info] Visualizing   : {len(vis_ds)} samples')
    print(f'[Info] Output dir    : {args.output_dir}')

    model = load_model(args.weight_path, device=args.device, use_dp=args.use_dp)

    with torch.no_grad():
        for idx, (images, targets, file_names) in enumerate(loader):
            images = images.to(args.device, non_blocking=True)

            outputs = model(images)
            anomaly_score = extract_anomaly_score(outputs)

            if anomaly_score.shape[-2:] != targets.shape[-2:]:
                anomaly_score = F.interpolate(
                    anomaly_score.unsqueeze(1),
                    size=targets.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                ).squeeze(1)

            image_np = denormalize_image(images[0].cpu())
            score_np = anomaly_score[0].cpu().numpy()

            score_vis = prepare_score_map_for_vis(
                score_np,
                low_percentile=args.low_percentile,
                high_percentile=args.high_percentile,
                gamma=args.gamma,
                suppress_threshold=args.suppress_threshold,
                smooth_kernel=args.smooth_kernel,
                invert_score=args.invert_score
            )

            save_name = f'{idx:02d}_{file_names[0]}.png'
            save_path = os.path.join(args.output_dir, save_name)

            title = f'{args.dataset} | {file_names[0]}'
            save_visualization(
                image_np=image_np,
                score_vis=score_vis,
                save_path=save_path,
                title=title,
                cmap=args.cmap,
                max_alpha=args.max_alpha,
                show_title=args.show_title
            )

            print(f'[Saved] {save_path}')


if __name__ == '__main__':
    main()