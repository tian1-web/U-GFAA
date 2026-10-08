import glob
import os
from collections import namedtuple
from typing import Any, Callable, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import numpy as np
import torchvision.transforms as standard_transforms
from lib.utils.img_utils import *
from lib.utils.utils import *
from typing import List


class Fishyscapes(Dataset):
    FishyscapesClass = namedtuple('FishyscapesClass', ['name', 'id', 'train_id', 'hasinstances',
                                                       'ignoreineval', 'color'])
    labels = [
        FishyscapesClass('in-distribution', 0, 0, False, False, (144, 238, 144)),
        FishyscapesClass('out-distribution', 2, 1, False, False, (255, 102, 102)),
        FishyscapesClass('ignore', 1, 255, False, True, (0, 0, 0)),
    ]

    train_id_in = 0
    train_id_out = 1

    def __init__(self, split='Static', root="/root/autodl-tmp/datasets/fishyscapes", transform=None):
        self.transform = transform
        self.root = root
        self.split = split
        self.images = []
        self.targets = []

        subset_root = os.path.join(self.root, self.split)
        image_dir = os.path.join(subset_root, 'original')
        target_dir = os.path.join(subset_root, 'labels')

        if os.path.exists(image_dir):
            for filename in sorted(os.listdir(image_dir)):
                if filename.endswith('.png'):
                    self.images.append(os.path.join(image_dir, filename))
                    self.targets.append(os.path.join(target_dir, filename))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = Image.open(self.images[i]).convert('RGB')
        target = Image.open(self.targets[i]).convert('L')

        target_np = np.array(target)
        binary_mask = np.zeros_like(target_np)
        binary_mask[target_np == 1] = 1
        target = Image.fromarray(binary_mask.astype(np.uint8), 'L')

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def __repr__(self):
        return f"Fishyscapes Dataset:\n  Split: {self.split}\n  Number of images: {len(self.images)}"


class RoadAnomaly(Dataset):
    RoadAnomaly_class = namedtuple('RoadAnomalyClass', ['name', 'id', 'train_id', 'hasinstances',
                                                        'ignoreineval', 'color'])
    labels = [
        RoadAnomaly_class('in-distribution', 0, 0, False, False, (144, 238, 144)),
        RoadAnomaly_class('out-distribution', 1, 1, False, False, (255, 102, 102)),
    ]

    train_id_in = 0
    train_id_out = 1
    num_eval_classes = 19
    label_id_to_name = {label.id: label.name for label in labels}
    train_id_to_name = {label.train_id: label.name for label in labels}
    trainid_to_color = {label.train_id: label.color for label in labels}
    label_name_to_id = {label.name: label.id for label in labels}

    def __init__(self, root='/root/shared-nvme/datasets/road_anomaly', transform=None):
        self.transform = transform
        self.root = root
        self.images = []
        self.targets = []

        img_dir = os.path.join(root, 'original')
        if os.path.exists(img_dir):
            filenames = sorted(os.listdir(img_dir))
            for filename in filenames:
                if os.path.splitext(filename)[1].lower() == '.jpg':
                    f_name = os.path.splitext(filename)[0]
                    filename_base_img = os.path.join("original", f_name)
                    filename_base_labels = os.path.join("labels", f_name)

                    self.images.append(os.path.join(self.root, filename_base_img + '.jpg'))
                    self.targets.append(os.path.join(self.root, filename_base_labels + '.png'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = Image.open(self.images[i]).convert('RGB')
        target = Image.open(self.targets[i]).convert('L')

        if self.transform is not None:
            image, target = self.transform(image, target)

        f_name = os.path.splitext(os.path.basename(self.images[i]))[0]
        return image, target, f_name

    def __repr__(self):
        return f"RoadAnomaly Dataset: {len(self.images)} images"


def _collect_eval_pairs(root: str):
    """
    通用收集函数：
    - images/ 下自动兼容 .jpg/.jpeg/.png/.webp
    - 标签固定对应 labels_masks/<basename>_labels_semantic.png
    """
    images = []
    targets = []

    img_dir = os.path.join(root, 'images')
    label_dir = os.path.join(root, 'labels_masks')

    if not os.path.isdir(img_dir):
        return images, targets
    if not os.path.isdir(label_dir):
        return images, targets

    valid_exts = {'.jpg', '.jpeg', '.png', '.webp'}

    for filename in sorted(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, filename)

        if not os.path.isfile(img_path):
            continue

        ext = os.path.splitext(filename)[1].lower()
        if ext not in valid_exts:
            continue

        f_name = os.path.splitext(filename)[0]
        target_path = os.path.join(label_dir, f_name + '_labels_semantic.png')

        if not os.path.isfile(target_path):
            continue

        images.append(img_path)
        targets.append(target_path)

    return images, targets


class RoadAnomaly21(Dataset):
    RoadAnomaly_class = namedtuple('RoadAnomalyClass', ['name', 'id', 'train_id', 'hasinstances',
                                                        'ignoreineval', 'color'])
    labels = [
        RoadAnomaly_class('in-distribution', 0, 0, False, False, (144, 238, 144)),
        RoadAnomaly_class('out-distribution', 1, 1, False, False, (255, 102, 102)),
    ]

    train_id_in = 0
    train_id_out = 1
    train_id_ignore = 255
    num_eval_classes = 19
    label_id_to_name = {label.id: label.name for label in labels}
    train_id_to_name = {label.train_id: label.name for label in labels}
    trainid_to_color = {label.train_id: label.color for label in labels}
    label_name_to_id = {label.name: label.id for label in labels}

    def __init__(self, root='/root/shared-nvme/datasets/segment_me/dataset_AnomalyTrack', transform=None):
        self.transform = transform
        self.root = root
        self.images, self.targets = _collect_eval_pairs(self.root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = Image.open(self.images[i]).convert('RGB')
        if os.path.exists(self.targets[i]):
            target = Image.open(self.targets[i]).convert('L')
        else:
            image_np = np.array(image)
            target = np.ones_like(image_np)[:, :, 0] * 255
            target = Image.fromarray(target.astype(np.uint8), 'L')

        if self.transform is not None:
            image, target = self.transform(image, target)

        f_name = os.path.splitext(os.path.basename(self.images[i]))[0]
        return image, target, f_name

    def __repr__(self):
        return f"RoadAnomaly21 Dataset: {len(self.images)} images"


class RoadObstacle21(Dataset):
    RoadAnomaly_class = namedtuple('RoadAnomalyClass', ['name', 'id', 'train_id', 'hasinstances',
                                                        'ignoreineval', 'color'])
    labels = [
        RoadAnomaly_class('in-distribution', 0, 0, False, False, (144, 238, 144)),
        RoadAnomaly_class('out-distribution', 1, 1, False, False, (255, 102, 102)),
    ]

    train_id_in = 0
    train_id_out = 1
    train_id_ignore = 255
    num_eval_classes = 19
    label_id_to_name = {label.id: label.name for label in labels}
    train_id_to_name = {label.train_id: label.name for label in labels}
    trainid_to_color = {label.train_id: label.color for label in labels}
    label_name_to_id = {label.name: label.id for label in labels}

    def __init__(self, root='/root/shared-nvme/datasets/segment_me/dataset_ObstacleTrack', transform=None, no_void=False):
        self.transform = transform
        self.root = root
        self.no_void = no_void
        self.images, self.targets = _collect_eval_pairs(self.root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = Image.open(self.images[i]).convert('RGB')
        if os.path.exists(self.targets[i]):
            target = Image.open(self.targets[i]).convert('L')
        else:
            image_np = np.array(image)
            target = np.ones_like(image_np)[:, :, 0] * 255
            target = Image.fromarray(target.astype(np.uint8), 'L')

        if self.transform is not None:
            image, target = self.transform(image, target)

        if self.no_void:
            target = np.array(target, dtype=np.uint8)
            target[target == self.train_id_ignore] = self.train_id_in
            target = Image.fromarray(target.astype(np.uint8), 'L')

        f_name = os.path.splitext(os.path.basename(self.images[i]))[0]
        return image, target, f_name

    def __repr__(self):
        return f"RoadObstacle21 Dataset: {len(self.images)} images"


class MUAD(Dataset):
    def __init__(self, root='/root/shared-nvme/datasets/MUAD_challenge/test_sets/test_OOD', transform=None):
        super(MUAD, self).__init__()
        self.transform = transform
        self.root = root
        self.img_root = os.path.join(self.root, 'leftImg8bit')
        self.gt_root = os.path.join(self.root, 'leftLabel')

        self.images = []
        if os.path.exists(self.img_root):
            self.images = sorted(glob.glob(os.path.join(self.img_root, "*.png")))

        self.f_names = [os.path.splitext(os.path.basename(img))[0] for img in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_dir = self.images[i]
        f_name = self.f_names[i]
        gt_dir = img_dir.replace("leftImg8bit", "leftLabel")
        image = Image.open(img_dir).convert('RGB')
        target = Image.open(gt_dir).convert('L')

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target, f_name