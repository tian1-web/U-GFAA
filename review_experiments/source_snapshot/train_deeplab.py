import datetime
import logging
import os
import time
import random # [新增] 用于概率控制
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from torch import nn
# [修改] 改名为 F_nn 以避免与 torchvision 的 F 冲突
import torch.nn.functional as F_nn  
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from lib.configs.parse_arg import opt, args
from lib.dataset import *
from lib.utils.img_utils import *
from lib.network.deepv3 import DeepWV3Plus
from lib.utils.metric import *
import lib.loss as loss_module
from lib.utils import random_init

# 导入 GPU 端频域增强函数
from lib.utils.freq_aug import uncertainty_aware_amp_mix

torch.backends.cudnn.benchmark = True


class TrainDeepLabOOD:
    """Trainer class for DeepLabV3+ with Out-of-Distribution detection."""
    
    def __init__(self) -> None:
        """Initialize the trainer with logging, datasets, model, and loss."""
        super().__init__()
        self.log_init()
        # 初始化 best 字典，分别记录三个数据集的 AUPRC
        self.best: Dict[str, float] = {
            'RoadAnomaly': -1.0,
            'RoadAnomaly21': -1.0,
            'RoadObstacle21': -1.0
        }
        self.criterion = self.build_loss()
        self.build_dataset()
        self.model = self.build_model(weight_path=args.weight_path)
        self.since = time.time()

    def build_dataset(self) -> None:
        """Initialize datasets and transformations for training and validation."""
        train_tf = Compose([
            ToTensor(),
            RandCrop(size=(opt.data.crop_size[0], opt.data.crop_size[1])),
            Normalize(mean=opt.data.mean, std=opt.data.std)
        ])
        
        test_tf = Compose([
            ToTensor(),
            Normalize(mean=opt.data.mean, std=opt.data.std),
        ])

        # [设置] 这里的 freq_aug 设为 False
        # 因为我们已经在下面的 train 循环中实现了 GPU 端的 U-SAFA 增强
        train_ds = DiverseCityscapes(
            split="train",
            transform=train_tf,
            anomaly_mix=opt.data.anomaly_mix,
            mixup=opt.data.mixup,
            freq_aug=False 
        )

        val_datasets = {
            'RoadAnomaly': RoadAnomaly(transform=test_tf),
            'RoadAnomaly21': RoadAnomaly21(transform=test_tf),
            'RoadObstacle21': RoadObstacle21(transform=test_tf)
        }

        self.data_loaders = {
            'train': DataLoader(
                train_ds,
                batch_size=opt.train.train_batch,
                drop_last=True,
                num_workers=opt.data.num_workers,
                shuffle=True,
                pin_memory=True
            )
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

    def build_model(self, class_num: int = 19, parallel: bool = True, 
                   weight_path: str = '') -> nn.Module:
        model = DeepWV3Plus(class_num)
        if parallel:
            model = nn.DataParallel(model)

        if not weight_path:
            self.logger.warning(
                "Using pretrained model trained in closed world without OOD. "
                "Please download the model and set weight_path in config file."
            )
            return model.cuda()

        state_dict = torch.load(weight_path)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        self.logger.info(f"Missing keys: {missing_keys}")
        self.logger.info(f"Unexpected keys: {unexpected_keys}")

        model.module.uncertainty_func_init()
        return model.cuda()

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
        self.logger.warning(
            f"Change trainable_params_name from {opt.model.trainable_params_name} "
            f"to {opt.model.trainable_params_name_update}"
        )
        self.logger.warning(
            f"Change lr from {opt.train.lr} to {opt.train.lr_update}"
        )

        opt.model.trainable_params_name = opt.model.trainable_params_name_update
        opt.train.lr = opt.train.lr_update
        
        params, names = self.configure_trainable_params()
        self.logger.warning(f'Trainable Params: {names}')
        self.optimizer = self.build_optimizer(params, opt.train.lr)

    def build_loss(self) -> nn.Module:
        Criterion = getattr(loss_module, opt.loss.name)
        return Criterion(opt.loss.params)

    def train(self) -> None:
        """Main training loop with U-SAFA and Consistency Loss."""
        self.writer = SummaryWriter(opt.tb_dir)
        params, names = self.configure_trainable_params()
        self.logger.warning(f'Trainable Params: {names}')
        self.optimizer = self.build_optimizer(params, opt.train.lr)
        
        loss_meter = MultiRunningMeter()
        self.model.train()

        # [超参设置]
        lambda_logit = 1.0 
        lambda_feat = 0.1  
        
        # 初始混合强度
        usafa_strength = 0.6 

        for epoch in range(args.start_epoch, opt.train.n_epochs):
            if epoch == opt.train.warmup_epoch:
                self.update_trainable_params()

            # [可选] 动态调整强度 (Curriculum Learning)
            # if epoch > 30: usafa_strength = 0.4

            for id_data, data in enumerate(self.data_loaders['train']):
                img, target = data[0].cuda(), data[1].cuda().long()
                div_img, div_target = data[2].cuda(), data[3].cuda().long()
                
                # 1. Clean Pass (原图前向传播)
                anomaly_score_clean, logit_clean, features_clean = self.model(img)
                
                # 2. Base Loss (基础分割 + OOD Loss)
                loss_clean = self.criterion(
                    logits=logit_clean, 
                    anomaly_score=anomaly_score_clean, 
                    targets=target,
                    freq_features=features_clean,
                    input_imgs=img
                ).mean()
                
                # 3. 数据增强策略选择 (Hybrid Strategy)
                # div_img (ControlNet生成) 作为风格参考源
                with torch.no_grad():
                    p = random.random()
                    
                    if p < 0.5:
                        # === 策略 A: U-SAFA (局部混合) - 50% ===
                        # 针对 RoadAnomaly (难样本/大偏移)
                        
                        # 归一化不确定性图 -> Mask
                        u_map = anomaly_score_clean.detach().unsqueeze(1)
                        B_size = u_map.shape[0]
                        u_map_flat = u_map.view(B_size, -1)
                        u_min = u_map_flat.min(dim=1, keepdim=True)[0].view(B_size, 1, 1, 1)
                        u_max = u_map_flat.max(dim=1, keepdim=True)[0].view(B_size, 1, 1, 1)
                        u_mask = (u_map - u_min) / (u_max - u_min + 1e-6)
                        
                        # 执行局部混合 (内部会自动高斯模糊)
                        img_aug = uncertainty_aware_amp_mix(img, div_img, u_mask, strength=usafa_strength)
                        
                    elif p < 0.8:
                        # === 策略 B: Global Mix (全图混合) - 30% ===
                        # 针对 RoadObstacle21 (保持全局风格鲁棒性)
                        # 传入 None 作为 mask，触发全图随机混合逻辑
                        img_aug = uncertainty_aware_amp_mix(img, div_img, uncertainty_map=None)
                        
                    else:
                        # === 策略 C: Identity (保持原图) - 20% ===
                        # 巩固基础特征，防止过拟合
                        img_aug = img.clone()

                    # 数值保护
                    img_aug = torch.clamp(img_aug, img.min(), img.max())

                # 4. Augmented Pass (增强图前向传播)
                _, logit_aug, features_aug = self.model(img_aug)
                
                # 5. Consistency Loss (一致性损失)
                
                # 5.1 Logit Consistency
                loss_con_logit = F_nn.kl_div(
                    F_nn.log_softmax(logit_aug, dim=1),
                    F_nn.softmax(logit_clean.detach(), dim=1),
                    reduction='mean' 
                )
                
                # 5.2 Feature Consistency
                loss_con_feat = F_nn.mse_loss(
                    features_aug, 
                    features_clean.detach(),
                    reduction='mean'
                )
                
                # 6. Total Loss
                total_loss = loss_clean + lambda_logit * loss_con_logit + lambda_feat * loss_con_feat

                self.log_epoch({'loss': total_loss.item()}, f"{epoch}_{id_data}")

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            avg_terms = loss_meter.get_metric()
            loss_meter.reset()

            if epoch % 1 == 0:
                self.model.eval()
                for ds_name, val_loader in self.val_loaders.items():
                    self.logger.info(f"Start validating on {ds_name}...")
                    metrics = self.valid_batch(dl=val_loader)
                    
                    log_data = {f"{ds_name}_{k}": v for k, v in metrics.items()}
                    self.log_epoch(log_data, epoch, phase='val')
                    
                    current_auprc = metrics['AUPRC']
                    if current_auprc > self.best[ds_name]:
                        self.best[ds_name] = current_auprc
                        self.logger.warning(f'Update best model for {ds_name} (AUPRC: {current_auprc:.4f})')
                        self.update_best(current_auprc, save_name=f'{ds_name}')
                
                self.model.train()

    def valid_batch(self, dl: Optional[DataLoader] = None) -> Dict[str, float]:
        anomaly_score_list = []
        ood_gts_list = []
        
        with torch.no_grad():
            for data in dl:
                img, target = data[0].cuda(), data[1].cuda().long()
                anomaly_score, logit, _ = self.model(img) 
                
                ood_gts_list.extend(target.cpu().numpy())
                anomaly_score_list.extend(anomaly_score.cpu().numpy())

        ood_gts = np.array(ood_gts_list)
        anomaly_scores = np.array(anomaly_score_list)
        roc_auc, prc_auc, fpr95 = eval_ood_measure(anomaly_scores, ood_gts)

        return {
            'AUROC': roc_auc,
            'AUPRC': prc_auc,
            'FPR_TPR95': fpr95
        }

    def update_best(self, avg_term: float, save_name: str = '') -> None:
        save_path = f'{opt.model_dir}/{save_name}_best_model.pth'
        torch.save(
            self.model.state_dict(),
            save_path
        )
        self.logger.warning(f'{args.id} saved best model to {save_path}')

    def plot_curves_multi(self, data: Dict[str, Any], epoch: int, phase: str = 'train') -> None:
        group_name = 'epoch_verbose'
        for key, value in data.items():
            self.writer.add_scalar(
                f'{group_name}/{phase}_{key}',
                value,
                epoch
            )

    def log_init(self) -> None:
        """Initialize logging configuration."""
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        # 仅使用 StreamHandler，防止重复写入
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)

        formatter = logging.Formatter(
            '[%(asctime)s.%(msecs)03d] %(message)s',
            datefmt='%H:%M:%S'
        )
        ch.setFormatter(formatter)

        self.logger.handlers = [ch]

        self.logger.info(str(opt))
        self.logger.info(f'Time: {datetime.datetime.now()}')

    def log_epoch(self, data: Dict[str, float], epoch: int, phase: str = 'train') -> None:
        log_str = f'{phase} Epoch: {epoch} '
        log_str += ' '.join([f'{k}: {v:.4f}' for k, v in data.items()])
        logging.warning(log_str)

    def log_final(self) -> None:
        log_str = 'Best Results:\n'
        for ds_name, score in self.best.items():
             log_str += f'{ds_name} AUPRC: {score:.6f}\n'
        logging.warning(log_str)

        time_elapsed = time.time() - self.since
        logging.warning(
            'Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
                time_elapsed // 60 // 60,
                time_elapsed // 60 % 60,
                time_elapsed % 60
            )
        )


if __name__ == '__main__':
    random_init(args.seed)
    os.makedirs(opt.model_dir, exist_ok=True)
    
    if not args.weight_path:
        args.weight_path = '../pretrained_model/DeepLabV3+_WideResNet38_baseline.pth'
        print(f"Load {args.weight_path}")

    ood = TrainDeepLabOOD()
    run_fn = getattr(ood, args.run)
    run_fn()