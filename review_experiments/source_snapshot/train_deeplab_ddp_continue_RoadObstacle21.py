import datetime
import logging
import os
import time
import random
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F_nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter

from lib.configs.parse_arg import opt, args
from lib.dataset import *
from lib.utils.img_utils import *
from lib.network.deepv3 import DeepWV3Plus
from lib.utils.metric import *
import lib.loss as loss_module
from lib.utils import random_init
from lib.utils.freq_aug import uncertainty_aware_amp_mix

torch.backends.cudnn.benchmark = True


class TrainDeepLabOOD:
    def __init__(self) -> None:
        self.init_distributed()

        if self.is_main_process():
            self.log_init()
            self.best: Dict[str, float] = {
                'RoadAnomaly': -1.0,
                'RoadAnomaly21': -1.0,
                'RoadObstacle21': -1.0
            }
            self.writer = SummaryWriter(opt.tb_dir)

        self.criterion = self.build_loss()
        self.build_dataset()
        self.model = self.build_model(weight_path=args.weight_path)

        self.scaler = GradScaler()
        self.since = time.time()

    def init_distributed(self):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    def is_main_process(self):
        return dist.get_rank() == 0

    def build_dataset(self) -> None:
        train_tf = Compose([
            ToTensor(),
            RandCrop(size=(opt.data.crop_size[0], opt.data.crop_size[1])),
            Normalize(mean=opt.data.mean, std=opt.data.std)
        ])

        test_tf = Compose([
            ToTensor(),
            Normalize(mean=opt.data.mean, std=opt.data.std)
        ])

        train_ds = DiverseCityscapes(
            split="train",
            transform=train_tf,
            anomaly_mix=opt.data.anomaly_mix,
            mixup=opt.data.mixup,
            freq_aug=False
        )

        self.train_sampler = DistributedSampler(train_ds, shuffle=True)

        self.data_loaders = {
            'train': DataLoader(
                train_ds,
                batch_size=opt.train.train_batch,
                drop_last=True,
                num_workers=opt.data.num_workers,
                sampler=self.train_sampler,
                shuffle=False,
                pin_memory=True
            )
        }

        if self.is_main_process():
            val_datasets = {
                'RoadAnomaly': RoadAnomaly(transform=test_tf),
                'RoadAnomaly21': RoadAnomaly21(transform=test_tf),
                'RoadObstacle21': RoadObstacle21(transform=test_tf)
            }

            self.val_loaders = {}
            for name, ds in val_datasets.items():
                self.val_loaders[name] = DataLoader(
                    ds,
                    batch_size=opt.train.valid_batch,
                    drop_last=False,
                    shuffle=False,
                    num_workers=2
                )

    def build_model(self, class_num: int = 19, parallel: bool = True, weight_path: str = '') -> nn.Module:
        model = DeepWV3Plus(class_num)

        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if weight_path:
            state_dict = torch.load(weight_path, map_location='cpu')

            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            new_state_dict = {
                k.replace('module.', ''): v
                for k, v in state_dict.items()
            }

            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

            if self.is_main_process():
                self.logger.warning(f'Loaded weights from: {weight_path}')
                if missing_keys:
                    self.logger.warning(f'Missing keys: {missing_keys}')
                if unexpected_keys:
                    self.logger.warning(f'Unexpected keys: {unexpected_keys}')

        # 只有从 Cityscapes/闭集预训练权重开始训练时，才初始化 ood_head。
        # 从已经训练好的 U-GFAA checkpoint 续训时，不能重新初始化 ood_head。
        wp = str(weight_path).lower() if weight_path else ""
        if (not weight_path) or ("cityscapes" in wp) or ("pretrained" in wp):
            model.uncertainty_func_init()

        model = model.cuda()

        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True
        )

        return model

    def configure_trainable_params(self) -> Tuple[List[torch.Tensor], List[str]]:
        params = []
        names = []

        trainable_names = opt.model.trainable_params_name

        if 'freq_enhancement' not in trainable_names:
            trainable_names.append('freq_enhancement')

        for name, param in self.model.named_parameters():
            if any(i in name for i in trainable_names):
                param.requires_grad = True
                params.append(param)
                names.append(name)
            else:
                param.requires_grad = False

        return params, names

    def build_optimizer(self, params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=opt.train.weight_decay
        )

    def update_trainable_params(self) -> None:
        opt.model.trainable_params_name = opt.model.trainable_params_name_update
        opt.train.lr = opt.train.lr_update

        params, names = self.configure_trainable_params()

        if self.is_main_process():
            self.logger.warning(f'Update Params & LR: {names}')

        self.optimizer = self.build_optimizer(params, opt.train.lr)

    def build_loss(self) -> nn.Module:
        Criterion = getattr(loss_module, opt.loss.name)
        return Criterion(opt.loss.params)

    def train(self) -> None:
        params, names = self.configure_trainable_params()

        if self.is_main_process():
            self.logger.warning(f'Trainable Params: {names}')

        self.optimizer = self.build_optimizer(params, opt.train.lr)

        # 当时最佳配置：logit consistency 权重为 1.0，feature consistency 权重为 0.1
        lambda_logit = 1.0
        lambda_feat = 0.1

        # 物理 Batch=2，双卡，梯度累加 4 次，逻辑 Batch=16
        accumulation_steps = 4

        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(args.start_epoch, opt.train.n_epochs):
            self.train_sampler.set_epoch(epoch)

            if epoch == opt.train.warmup_epoch:
                self.update_trainable_params()
                self.optimizer.zero_grad(set_to_none=True)

            # RoadObstacle21 短程续训策略：
            # RoadObstacle21 best 出现在 epoch 32，因此从 epoch 32 的 best checkpoint 出发。
            # 32-36 轮：极弱增强 0.02，尝试轻微提升小障碍物泛化。
            # 36-40 轮：关闭增强，稳定微调，避免破坏已有高 AUPRC。
            if epoch < 36:
                current_strength = 0.02
            else:
                current_strength = 0.0

            if self.is_main_process():
                self.logger.warning(f"Epoch {epoch} | U-SAFA Strength: {current_strength:.4f}")

            self.model.train()

            for id_data, data in enumerate(self.data_loaders['train']):
                img = data[0].cuda(non_blocking=True)
                target = data[1].cuda(non_blocking=True).long()
                div_img = data[2].cuda(non_blocking=True)
                div_target = data[3].cuda(non_blocking=True).long()

                img_aug = None

                with autocast():
                    anomaly_score_clean, logit_clean, features_clean = self.model(img)

                    loss_clean = self.criterion(
                        logits=logit_clean,
                        anomaly_score=anomaly_score_clean,
                        targets=target,
                        freq_features=features_clean,
                        input_imgs=img
                    ).mean()

                    if current_strength > 0.01:
                        with torch.no_grad():
                            p = random.random()

                            # 当时最佳增强策略：
                            # 40% U-SAFA 局部不确定性引导混合
                            # 30% Global 全图混合
                            # 30% Identity 原图
                            if p < 0.4:
                                u_map = anomaly_score_clean.detach().unsqueeze(1)

                                B_size = u_map.shape[0]
                                u_map_flat = u_map.view(B_size, -1)
                                u_min = u_map_flat.min(dim=1, keepdim=True)[0].view(B_size, 1, 1, 1)
                                u_max = u_map_flat.max(dim=1, keepdim=True)[0].view(B_size, 1, 1, 1)

                                u_mask = (u_map - u_min) / (u_max - u_min + 1e-6)

                                img_aug = uncertainty_aware_amp_mix(
                                    img,
                                    div_img,
                                    u_mask,
                                    strength=current_strength
                                )

                                del u_map, u_map_flat, u_min, u_max, u_mask

                            elif p < 0.7:
                                img_aug = uncertainty_aware_amp_mix(
                                    img,
                                    div_img,
                                    uncertainty_map=None,
                                    strength=current_strength * 0.8
                                )

                            else:
                                img_aug = img.clone()

                            img_aug = torch.clamp(img_aug, img.min(), img.max())

                        _, logit_aug, features_aug = self.model(img_aug)

                        loss_con_logit = F_nn.kl_div(
                            F_nn.log_softmax(logit_aug, dim=1),
                            F_nn.softmax(logit_clean.detach(), dim=1),
                            reduction='mean'
                        )

                        loss_con_feat = F_nn.mse_loss(
                            features_aug,
                            features_clean.detach(),
                            reduction='mean'
                        )

                        # 当时最佳配置：一致性损失随 0.6 基准强度同步衰减
                        weight_factor = current_strength / 0.02

                        total_loss = loss_clean + (
                            lambda_logit * loss_con_logit +
                            lambda_feat * loss_con_feat
                        ) * weight_factor

                    else:
                        total_loss = loss_clean

                    total_loss = total_loss / accumulation_steps

                self.scaler.scale(total_loss).backward()

                if self.is_main_process():
                    self.log_epoch(
                        {'loss': total_loss.item() * accumulation_steps},
                        f"{epoch}_{id_data}"
                    )

                if (id_data + 1) % accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                    del anomaly_score_clean, logit_clean, features_clean, total_loss

                    if img_aug is not None:
                        del img_aug

                    if 'logit_aug' in locals():
                        del logit_aug

                    if 'features_aug' in locals():
                        del features_aug

                    if 'loss_con_logit' in locals():
                        del loss_con_logit

                    if 'loss_con_feat' in locals():
                        del loss_con_feat

            if self.is_main_process() and epoch % 1 == 0:
                self.model.eval()

                for ds_name, val_loader in self.val_loaders.items():
                    self.logger.warning(f"Start validating on {ds_name}...")

                    metrics = self.valid_batch(dl=val_loader)
                    log_data = {
                        f"{ds_name}_{k}": v
                        for k, v in metrics.items()
                    }

                    self.log_epoch(log_data, epoch, phase='val')

                    current_auprc = metrics['AUPRC']

                    if current_auprc > self.best[ds_name]:
                        self.best[ds_name] = current_auprc
                        self.logger.warning(
                            f'Update best model for {ds_name} '
                            f'(AUPRC: {current_auprc:.4f})'
                        )
                        self.update_best(current_auprc, save_name=f'{ds_name}')

                dist.barrier()

            else:
                dist.barrier()

    def valid_batch(self, dl: Optional[DataLoader] = None) -> Dict[str, float]:
        anomaly_score_list = []
        ood_gts_list = []

        with torch.no_grad():
            for data in dl:
                img = data[0].cuda(non_blocking=True)
                target = data[1].cuda(non_blocking=True).long()

                if isinstance(self.model, nn.parallel.DistributedDataParallel):
                    infer_model = self.model.module
                else:
                    infer_model = self.model

                anomaly_score, logit, _ = infer_model(img)

                ood_gts_list.extend(target.cpu().numpy())
                anomaly_score_list.extend(anomaly_score.cpu().numpy())

        roc_auc, prc_auc, fpr95 = eval_ood_measure(
            np.array(anomaly_score_list),
            np.array(ood_gts_list)
        )

        return {
            'AUROC': roc_auc,
            'AUPRC': prc_auc,
            'FPR_TPR95': fpr95
        }

    def update_best(self, avg_term: float, save_name: str = '') -> None:
        save_path = f'{opt.model_dir}/{save_name}_best_model.pth'

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), save_path)

        self.logger.warning(f'{args.id} saved best model to {save_path}')

    def log_init(self) -> None:
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)

        formatter = logging.Formatter(
            '[%(asctime)s.%(msecs)03d] %(message)s',
            datefmt='%H:%M:%S'
        )
        ch.setFormatter(formatter)

        self.logger.handlers = [ch]
        self.logger.info(str(opt))

    def log_epoch(self, data: Dict[str, float], epoch: int, phase: str = 'train') -> None:
        log_str = f'{phase} Epoch: {epoch} '
        log_str += ' '.join([f'{k}: {v:.4f}' for k, v in data.items()])
        logging.warning(log_str)


if __name__ == '__main__':
    random_init(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0:
        os.makedirs(opt.model_dir, exist_ok=True)

    if not args.weight_path:
        args.weight_path = '../pretrained_model/DeepLabV3+_WideResNet38_baseline.pth'

    ood = TrainDeepLabOOD()
    getattr(ood, args.run)()
PY