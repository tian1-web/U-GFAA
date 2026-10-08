import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms 
from torch.cuda.amp import autocast

# ===================================================================
# ===== 你的新损失函数定义 (START) ===================================
# ===================================================================

def generate_high_freq_mask(feature_map):
    B, C, H, W = feature_map.shape
    y, x = torch.meshgrid(torch.arange(H, device=feature_map.device),
                          torch.arange(W, device=feature_map.device), indexing='ij')
    mask = (x + y).float() / (H + W - 2) if H > 1 and W > 1 else torch.zeros(H, W, device=feature_map.device)
    return mask.unsqueeze(0).unsqueeze(0) 

def frequency_consistency_loss(freq_enhanced_features, imgs):
    """计算特征图和下采样图像在高频区域的频域一致性。"""
    
    with autocast(enabled=False):
        # 1. 类型转换
        freq_feat_fp32 = freq_enhanced_features.float()
        imgs_fp32 = imgs.float()

        # 2. 空间尺寸对齐 (Downsample Image)
        imgs_fp32 = F.interpolate(imgs_fp32, size=freq_feat_fp32.shape[2:], mode='bilinear', align_corners=True)
        
        # 3. [核心修复] 通道对齐策略：通道压缩 (Channel Reduction)
        # 将特征图和图片都压缩成单通道，只比较"结构/能量"的一致性
        # 特征图：取通道平均 (B, 1, H, W)
        feat_reduced = torch.mean(freq_feat_fp32, dim=1, keepdim=True)
        # 原图：转为灰度图 (B, 1, H, W)
        # RGB -> Gray 公式: 0.299R + 0.587G + 0.114B
        if imgs_fp32.shape[1] == 3:
            img_reduced = (0.299 * imgs_fp32[:, 0:1, :, :] + 
                           0.587 * imgs_fp32[:, 1:2, :, :] + 
                           0.114 * imgs_fp32[:, 2:3, :, :])
        else:
            img_reduced = torch.mean(imgs_fp32, dim=1, keepdim=True)

        # 4. 生成掩码 (使用压缩后的特征图生成)
        high_freq_mask = generate_high_freq_mask(feat_reduced)
        
        # 5. FFT
        freq_fft = torch.fft.rfft2(feat_reduced * high_freq_mask, norm='ortho')
        img_fft = torch.fft.rfft2(img_reduced * high_freq_mask, norm='ortho')

        # 6. Loss
        B = freq_feat_fp32.shape[0]
        loss_real = 1 - F.cosine_similarity(freq_fft.real.view(B, -1), img_fft.real.view(B, -1), dim=-1).mean()
        loss_imag = 1 - F.cosine_similarity(freq_fft.imag.view(B, -1), img_fft.imag.view(B, -1), dim=-1).mean()
    
    return loss_real + loss_imag

def frequency_sparsity_loss(freq_enhanced_features):
    # 同样禁用 AMP
    with autocast(enabled=False):
        feat_float = freq_enhanced_features.float()
        freq_fft = torch.fft.rfft2(feat_float, norm='ortho')
        loss = torch.mean(torch.abs(freq_fft))
    return loss


# ===================================================================
# ===== 你的新损失函数定义 (END) =====================================
# ===================================================================


class RelContrastiveLoss(nn.Module):
    """Relative Contrastive Loss for anomaly detection.
    
    Args:
        param_dict (dict): Dictionary containing configuration parameters.
    """
    
    def __init__(self, param_dict):
        super().__init__()
        # Initialize base loss function
        self.nll_loss = nn.NLLLoss(reduction='none', ignore_index=255)
        
        # Contrastive learning parameters
        self.inoutaug_contras_margins_tri = param_dict.get('inoutaug_contras_margins_tri', None)
        self.sample_ratio = param_dict.get('sample_ratio', 1)
        
        # Sample selection parameters
        self.conduct_pixel_selection = param_dict.get('conduct_pixel_selection', False)
        self.selection_ratio = param_dict.get('selection_ratio', 1.0)
        
        # Loss weights
        self.ce_weights = param_dict.get('ce_weights', [1, 1])
        self.contras_weight = param_dict.get('contras_weight', 1.0)

        # Class IDs for in-distribution and void classes.
        self.in_id = 99
        self.void_id = 255

    def forward(self, logits, anomaly_score, targets):
        """Compute the relative contrastive loss."""
        ood_mask = (targets > self.in_id) & (targets != self.void_id)
        in_mask = (targets < self.in_id)
        
        loss = 0.0
        batch_size = logits.shape[0]
        
        in_targets = targets.clone()
        in_targets[~in_mask] = 255
        in_mask_selected = in_mask.clone()
        
        ce_original = self.nll_loss(F.log_softmax(logits[:batch_size//2], dim=1), in_targets[:batch_size//2]).mean()
        
        if self.conduct_pixel_selection and 0.0 < self.selection_ratio < 1.0:
            ce_aug = self._compute_augmented_ce_with_selection(logits, in_targets, in_mask_selected, targets, batch_size)
        else:
            ce_aug = self.nll_loss(F.log_softmax(logits[batch_size//2:], dim=1), in_targets[batch_size//2:]).mean()
            ce_aug = torch.tensor(0.0, device=logits.device) if torch.isnan(ce_aug) else ce_aug
        
        ce_loss = (self.ce_weights[0] * ce_original + self.ce_weights[1] * ce_aug)
        loss += ce_loss
        
        in_mask_original, in_mask_aug = in_mask.clone(), in_mask.clone()
        in_mask_original[batch_size//2:] = False
        in_mask_aug[:batch_size//2] = False
        
        contrastive_loss = self._compute_contrastive_loss(anomaly_score, in_mask_original, in_mask_aug, ood_mask, batch_size)
        loss += self.contras_weight * contrastive_loss
        
        return loss
    
    def _compute_augmented_ce_with_selection(self, logits, in_targets, in_mask_selected, targets, batch_size):
        """Compute CE loss for augmented samples with pixel selection."""
        ce_aug = self.nll_loss(F.log_softmax(logits[batch_size//2:], dim=1), in_targets[batch_size//2:]).flatten()
        
        ce_aug_detach = ce_aug.detach()
        ce_aug_detach[in_targets[batch_size//2:].flatten() == 255] = float('inf')
        
        total_num = in_mask_selected[batch_size//2:].sum()
        select_num = int(self.selection_ratio * total_num)
        
        if select_num > 0:
            select_index = torch.topk(ce_aug_detach, select_num, largest=False)[1]
            ce_aug = ce_aug[select_index].mean()
            
            _, height, width = in_mask_selected.shape
            select_index_mask = torch.zeros_like(in_mask_selected[batch_size//2:]).flatten().bool()
            select_index_mask[select_index] = True
            select_index_mask = select_index_mask.reshape(batch_size//2, height, width)
            
            in_mask_selected[batch_size//2:][~select_index_mask] = False
            targets[batch_size//2:][~select_index_mask] = 255
        else:
            ce_aug = torch.tensor(0.0, device=logits.device)
            in_mask_selected[batch_size//2:] = False
            targets[batch_size//2:] = 255
        
        return ce_aug
    
    def _compute_contrastive_loss(self, anomaly_score, in_mask_original, in_mask_aug, ood_mask, batch_size):
        """Compute the three components of contrastive loss."""
        original_feature = anomaly_score[in_mask_original]
        aug_feature = anomaly_score[in_mask_aug]
        ood_feature = anomaly_score[ood_mask]
        
        num_samples = self._get_num_samples(in_mask_original, in_mask_aug, ood_mask)
        
        if num_samples > 0:
            original_feature = original_feature[torch.randperm(original_feature.shape[0])[:num_samples]]
            aug_feature = aug_feature[torch.randperm(aug_feature.shape[0])[:num_samples]]
            ood_feature = ood_feature[torch.randperm(ood_feature.shape[0])[:num_samples]]

            contras_original = F.relu(original_feature + self.inoutaug_contras_margins_tri[0] - ood_feature).mean()
            contras_aug = F.relu(aug_feature + self.inoutaug_contras_margins_tri[1] - ood_feature).mean()
        else:
            contras_original = torch.tensor(0.0, device=anomaly_score.device)
            contras_aug = torch.tensor(0.0, device=anomaly_score.device)
        
        same_in_mask = in_mask_original[:batch_size//2] & in_mask_aug[batch_size//2:]
        contras_in = F.relu(anomaly_score[batch_size//2:] - anomaly_score[:batch_size//2] - self.inoutaug_contras_margins_tri[2])[same_in_mask].mean()
        
        return (contras_original + contras_aug + contras_in)
    
    def _get_num_samples(self, in_mask_original, in_mask_aug, ood_mask):
        """Calculate number of samples based on sampling ratio."""
        total_pixels = in_mask_original.shape[0] * in_mask_original.shape[1] * in_mask_original.shape[2]
        num_samples = int(total_pixels * self.sample_ratio)
        num_samples = min(num_samples, ood_mask.sum())
        num_samples = min(num_samples, in_mask_original.sum())
        num_samples = min(num_samples, in_mask_aug.sum())
        return num_samples

# ===================================================================
# ===== 新的组合损失类 (START) =======================================
# ===================================================================

class MyCustomLoss(nn.Module):
    """
    组合了 RelContrastiveLoss 和频域损失的自定义损失函数。
    """
    def __init__(self, param_dict):
        super().__init__()
        # 1. 实例化原始的损失函数
        self.original_loss = RelContrastiveLoss(param_dict)
        
        # 2. 从配置字典中获取新损失的权重
        self.lambda_consistency = param_dict.get('lambda_consistency', 0.1)
        self.lambda_sparsity = param_dict.get('lambda_sparsity', 0.01)

    def forward(self, logits, anomaly_score, targets, freq_features, input_imgs):
        """
        计算总损失。
        Args:
            logits (torch.Tensor): 模型的语义分割输出。
            anomaly_score (torch.Tensor): 模型的异常分数输出。
            targets (torch.Tensor): 真实标签。
            freq_features (torch.Tensor): 从频域模块输出的中间特征。
            input_imgs (torch.Tensor): 模型的原始输入图像。
        """
        # 1. 计算原始损失
        original_loss_val = self.original_loss(logits, anomaly_score, targets)
        
        # 2. 计算频率一致性损失
        consistency_loss_val = frequency_consistency_loss(freq_features, input_imgs)

        # 3. 计算频率稀疏性损失
        sparsity_loss_val = frequency_sparsity_loss(freq_features)

        # 4. 组合所有损失
        total_loss = (original_loss_val + 
                      self.lambda_consistency * consistency_loss_val + 
                      self.lambda_sparsity * sparsity_loss_val)
        
        # (可选) 打印各个损失项的值，方便调试
        # print(f"Original: {original_loss_val.item():.4f}, "
        #       f"Consistency: {consistency_loss_val.item():.4f}, "
        #       f"Sparsity: {sparsity_loss_val.item():.4f}")
        
        return total_loss

# ===================================================================
# ===== 新的组合损失类 (END) =========================================
# ===================================================================