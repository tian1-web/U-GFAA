import os
import sys
import argparse
from typing import Tuple

# =========================================================
# Step 1: parse OUR args first, before importing repo model
# =========================================================
pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument('--cfg', type=str, default='exps/DeepLab.yaml')
pre_parser.add_argument('--id', type=str, default='fishyscapes_eval')
pre_parser.add_argument('--test_dataset', type=str, required=True,
                        help='FishyscapesStatic or FishyscapesLostAndFound')
pre_parser.add_argument('--weight_path', type=str, required=True)
pre_parser.add_argument('--data_root', type=str, default='/root/shared-nvme/datasets/fishyscapes')
pre_parser.add_argument('--batch_size', type=int, default=1)
pre_parser.add_argument('--num_workers', type=int, default=0)
pre_parser.add_argument('--device', type=str, default='cuda')
pre_parser.add_argument('--use_dp', action='store_true')

args, _unknown = pre_parser.parse_known_args()

# =========================================================
# Step 2: rewrite sys.argv so repo parser only sees safe args
# =========================================================
sys.argv = [
    sys.argv[0],
    '--cfg', args.cfg,
    '--id', args.id,
    '--test_dataset', args.test_dataset,
    '--weight_path', args.weight_path,
]

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T

from lib.network.deepv3 import DeepWV3Plus
from lib.dataset.cityscapes import DiverseCityscapes
from lib.dataset.fishyscapes import Fishyscapes


class FishyscapesTestTransform:
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


def build_dataset(test_dataset: str, data_root: str, transform):
    name = test_dataset.lower()

    if name in ['fishyscapesstatic', 'static', 'fsstatic']:
        split = 'Static'
    elif name in ['fishyscapeslostandfound', 'lostandfound', 'fslostandfound', 'laf']:
        split = 'LostAndFound'
    else:
        raise ValueError(
            f'Unsupported test_dataset={test_dataset}. '
            f'Use FishyscapesStatic or FishyscapesLostAndFound.'
        )

    return Fishyscapes(
        split=split,
        root=data_root,
        transform=transform
    ), split


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


def extract_anomaly_score(outputs):
    """
    Automatically extract anomaly score from different model output formats.

    Supported:
    1) tensor of shape [B, H, W]
    2) tensor of shape [B, 1, H, W]
    3) tuple/list containing one of the above
    """
    if torch.is_tensor(outputs):
        if outputs.ndim == 3:
            return outputs
        if outputs.ndim == 4 and outputs.shape[1] == 1:
            return outputs.squeeze(1)
        raise RuntimeError(f'Unsupported tensor output shape for anomaly score: {outputs.shape}')

    if isinstance(outputs, (list, tuple)):
        # Prefer [B, H, W]
        for out in outputs:
            if torch.is_tensor(out) and out.ndim == 3:
                return out
        # Then [B, 1, H, W]
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

    raise RuntimeError(f'Unsupported model output type: {type(outputs)}')


def compute_ood_metrics(scores: np.ndarray, labels: np.ndarray):
    """
    labels:
        0 -> inlier
        1 -> OOD
    """
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)

    # FPR at 95% TPR
    idxs = np.where(tpr >= 0.95)[0]
    if len(idxs) == 0:
        fpr95 = 1.0
    else:
        fpr95 = fpr[idxs[0]]

    return auroc, auprc, fpr95


def evaluate(model, loader, device='cuda'):
    all_scores = []
    all_labels = []

    total_samples = len(loader.dataset)
    processed = 0

    with torch.no_grad():
        for images, targets, file_names in tqdm(loader, total=len(loader), desc='Testing Fishyscapes'):
            images = images.to(device, non_blocking=True)

            outputs = model(images)
            anomaly_score = extract_anomaly_score(outputs)

            # upsample if needed
            if anomaly_score.shape[-2:] != targets.shape[-2:]:
                anomaly_score = F.interpolate(
                    anomaly_score.unsqueeze(1),
                    size=targets.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                ).squeeze(1)

            anomaly_score = anomaly_score.cpu().numpy()
            targets = targets.numpy()

            for score_map, target_map, file_name in zip(anomaly_score, targets, file_names):
                valid_mask = target_map != 255
                valid_scores = score_map[valid_mask]
                valid_labels = target_map[valid_mask]

                all_scores.append(valid_scores.reshape(-1))
                all_labels.append(valid_labels.reshape(-1))

                processed += 1
                if processed % 50 == 0 or processed == total_samples:
                    print(f'[Progress] {processed}/{total_samples} images done, current file = {file_name}', flush=True)

    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    auroc, auprc, fpr95 = compute_ood_metrics(all_scores, all_labels)

    print('\n===== Fishyscapes OOD Diagnostics =====')
    print(f'Valid pixels       : {all_labels.shape[0]}')
    print(f'Label values seen  : {sorted(np.unique(all_labels).tolist())}')
    print(f'Score min / max    : {all_scores.min():.6f} / {all_scores.max():.6f}')

    return {
        'AUROC': float(auroc),
        'AUPRC': float(auprc),
        'FPR95': float(fpr95),
    }


def main():
    torch.backends.cudnn.benchmark = True

    transform = FishyscapesTestTransform()
    dataset, split = build_dataset(args.test_dataset, args.data_root, transform)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f'[Info] Fishyscapes split  : {split}')
    print(f'[Info] Data root          : {args.data_root}')
    print(f'[Info] Found {len(dataset)} samples.')

    model = load_model(args.weight_path, device=args.device, use_dp=args.use_dp)
    metrics = evaluate(model, loader, device=args.device)

    print('\n===== Fishyscapes OOD Results =====')
    print(f'AUROC : {metrics["AUROC"] * 100:.2f}')
    print(f'AUPRC : {metrics["AUPRC"] * 100:.2f}')
    print(f'FPR95 : {metrics["FPR95"] * 100:.2f}')


if __name__ == '__main__':
    main()