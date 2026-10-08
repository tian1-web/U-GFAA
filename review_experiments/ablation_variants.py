"""Explicit, named interventions around the supplied implementation.

No variant rewires FEM into the prediction path. Incremental rows therefore test
the supplied auxiliary-FEM implementation, with its limitations, honestly.
"""
import contextlib
from dataclasses import asdict, dataclass, replace
import importlib.util
import random
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import runtime
from training_budget import source_epoch


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    baseline: bool = False
    usafa: bool = True
    fem: bool = True
    consistency: bool = True
    schedule: str = 'coupled'
    mask: str = 'uncertainty'
    phase_lock: bool = True
    sigma: float = 2.0
    fda_beta: float = 0.01
    margin_factor: float = 1.0
    lambda_logit: float = 1.0
    lambda_feature: float = 0.1

    def record(self):
        info = asdict(self)
        info.update(
            mask_comparison_scope='Only the original 40% uncertainty-guided branch is replaced; global 30% and identity 30% are retained.',
            uniform_mask='Constant per image equal to the mean of the smoothed uncertainty mixing mask.',
            random_mask='Smoothed random field rank-mapped to the exact histogram of the smoothed uncertainty mask.',
            fda_like='Weighted low-frequency rectangular amplitude mixing, beta=0.01; not a reproduction of the complete FDA method; mean mixing budget differs.',
            unlocked_phase='Circular shortest-angle phase interpolation using the same frequency mixing coefficients; applied to both non-identity branches.',
            incremental_scope='Source clean-only RCL loss retained. Without consistency, the augmented forward affects running BN statistics but supplies no augmented task gradient. FEM remains an auxiliary output.',
            baseline_scope='Upstream MultiShiftSeg model, paired clean/generated batch and RCL, upstream learning rates; AMP and accumulation adapt the batch to this GPU.')
        return info


SPECS = [
    Variant('full_source', 'Supplied complete U-GFAA implementation, control.'),
    Variant('multishiftseg', 'Nearest baseline from the supplied Git HEAD.', baseline=True,
            usafa=False, fem=False, consistency=False),
    Variant('curriculum_fixed', 'No curriculum: constant augmentation strength 0.6 and consistency weight 1.', schedule='fixed'),
    Variant('curriculum_strength_only', 'Scheduled augmentation strength; fixed consistency weight 1.', schedule='strength_only'),
    Variant('curriculum_weight_only', 'Fixed augmentation strength 0.6; scheduled consistency weight.', schedule='weight_only'),
    Variant('uniform_mask', 'Mean-matched constant mixing mask in the guided branch.', mask='uniform'),
    Variant('random_mask', 'Random smooth mask with an exactly matched coefficient histogram.', mask='random'),
    Variant('fda_like', 'Low-frequency mixing replaces the guided branch.', mask='fda'),
    Variant('no_phase_lock', 'Mix phases as well as amplitudes using the same coefficient.', phase_lock=False),
    Variant('incremental_base', 'DeepLabV3+ source clean task loss; no U-SAFA, FEM or cross-view consistency.',
            usafa=False, fem=False, consistency=False),
    Variant('incremental_usafa', 'Add the source U-SAFA augmented forward to the base.', fem=False, consistency=False),
    Variant('incremental_fem', 'Add auxiliary FEM and its frequency losses; no cross-view consistency.', consistency=False),
    Variant('sigma_low', 'Gaussian sigma 1 instead of 2, kernel size remains 7.', sigma=1.0),
    Variant('sigma_high', 'Gaussian sigma 4 instead of 2, kernel size remains 7.', sigma=4.0),
    Variant('margin_low', 'All three source RCL margins multiplied by 0.5.', margin_factor=0.5),
    Variant('margin_high', 'All three source RCL margins multiplied by 2.', margin_factor=2.0),
    Variant('logit_weight_low', 'Logit consistency coefficient 0.5; feature coefficient 0.1.', lambda_logit=0.5),
    Variant('logit_weight_high', 'Logit consistency coefficient 2; feature coefficient 0.1.', lambda_logit=2.0),
    Variant('feature_weight_low', 'Feature consistency coefficient 0.05; logit coefficient 1.', lambda_feature=0.05),
    Variant('feature_weight_high', 'Feature consistency coefficient 0.2; logit coefficient 1.', lambda_feature=0.2),
]
VARIANTS = {v.name: v for v in SPECS}


def adjust_config(opt, spec):
    if spec.baseline:
        opt.train.lr = 1e-4
        opt.model.trainable_params_name = ['ood_head']
        opt.model.trainable_params_name_update = ['aspp', 'bot_fine', 'bot_aspp', 'ood_head']
        opt.loss.name = 'UpstreamRelContrastiveLossWithEmptySetGuard'
    elif not spec.fem:
        opt.model.trainable_params_name = [s for s in opt.model.trainable_params_name if s != 'freq_enhancement']
        opt.model.trainable_params_name_update = [s for s in opt.model.trainable_params_name_update if s != 'freq_enhancement']
        opt.loss.name = 'RelContrastiveLoss'
    opt.loss.params.inoutaug_contras_margins_tri = [v * spec.margin_factor for v in opt.loss.params.inoutaug_contras_margins_tri]


def load_upstream(relative, name):
    path = runtime.EXP / 'upstream_snapshot' / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def model_class(source_class, spec):
    if not spec.baseline:
        return source_class
    upstream = load_upstream('lib/network/deepv3/deepv3.py', 'lib.network.deepv3.upstream_baseline')

    class BaselineModel(upstream.DeepWV3Plus):
        def forward(self, image):
            score, logits = super().forward(image)
            return score, logits, None
    return BaselineModel


def finalize_model(model, meta, spec):
    if not spec.fem and hasattr(model, 'freq_enhancement'):
        model.freq_enhancement = nn.Identity()
    meta['executed_model_parameters'] = sum(p.numel() for p in model.parameters())
    meta['variant_model'] = spec.name
    return model, meta


def make_criterion(opt, spec):
    from lib.loss import MyCustomLoss, RelContrastiveLoss
    if spec.baseline:
        upstream = load_upstream('lib/loss.py', 'upstream_baseline_loss')

        class GuardedBaselineLoss(upstream.RelContrastiveLoss):
            """Only the undefined empty-sample means are guarded, without skipping a batch."""
            def _compute_contrastive_loss(self, score, original, aug, ood, batch_size):
                count = self._get_num_samples(original, aug, ood)
                same = original[:batch_size // 2] & aug[batch_size // 2:]
                if count > 0 and same.any():
                    return super()._compute_contrastive_loss(score, original, aug, ood, batch_size)
                # Retain the upstream RNG consumption even for zero-length samples.
                a, b, c = score[original], score[aug], score[ood]
                a = a[torch.randperm(a.shape[0])[:count]]
                b = b[torch.randperm(b.shape[0])[:count]]
                c = c[torch.randperm(c.shape[0])[:count]]
                zero = score.sum() * 0
                first = F.relu(a + self.inoutaug_contras_margins_tri[0] - c).mean() if count > 0 else zero
                second = F.relu(b + self.inoutaug_contras_margins_tri[1] - c).mean() if count > 0 else zero
                third = F.relu(score[batch_size // 2:] - score[:batch_size // 2]
                               - self.inoutaug_contras_margins_tri[2])[same].mean() if same.any() else zero
                return first + second + third
        return GuardedBaselineLoss(opt.loss.params)
    if spec.fem:
        return MyCustomLoss(opt.loss.params)

    class TaskOnlyLoss(RelContrastiveLoss):
        def forward(self, logits, anomaly_score, targets, **unused):
            return super().forward(logits, anomaly_score, targets)
    return TaskOnlyLoss(opt.loss.params)


def strengths(epoch, spec, total_epochs=49):
    scheduled = runtime.curriculum_strength(source_epoch(epoch, total_epochs))
    if not spec.usafa:
        return 0.0, 0.0
    if spec.schedule == 'coupled':
        return scheduled, scheduled / 0.6
    if spec.schedule == 'fixed':
        return 0.6, 1.0
    if spec.schedule == 'strength_only':
        return scheduled, 1.0
    if spec.schedule == 'weight_only':
        return 0.6, scheduled / 0.6
    raise ValueError(spec.schedule)


def frequency_mask(uncertainty, image_shape, strength, spec):
    height, width = image_shape[-2:]
    mask = F.interpolate(uncertainty, size=(height, width // 2 + 1), mode='bilinear')
    mask = TF.gaussian_blur(mask, kernel_size=7, sigma=spec.sigma)
    if spec.mask == 'uncertainty':
        return mask * strength
    if spec.mask == 'uniform':
        return mask.mean(dim=(-2, -1), keepdim=True).expand_as(mask) * strength
    if spec.mask == 'random':
        noise = TF.gaussian_blur(torch.rand_like(mask), kernel_size=7, sigma=spec.sigma)
        order = noise.flatten(1).argsort(dim=1)
        sorted_values = mask.flatten(1).sort(dim=1).values
        shuffled = torch.empty_like(sorted_values).scatter_(1, order, sorted_values)
        return shuffled.view_as(mask) * strength
    if spec.mask == 'fda':
        # A symmetric rectangle around DC, represented in rfft coordinates.
        b = int(min(height, width) * spec.fda_beta)
        row = torch.arange(height, device=mask.device)
        col = torch.arange(width // 2 + 1, device=mask.device)
        support = ((row <= b) | (row >= height - b))[:, None] & (col <= b)[None, :]
        return support.to(mask.dtype)[None, None].expand_as(mask) * strength
    raise ValueError(spec.mask)


def mix_variant(source, style, uncertainty, strength, spec):
    from lib.utils.freq_aug import uncertainty_aware_amp_mix
    if spec.phase_lock and (uncertainty is None or (spec.mask == 'uncertainty' and spec.sigma == 2.0)):
        return uncertainty_aware_amp_mix(source, style, uncertainty, strength=strength)
    src_fft = torch.fft.rfft2(source.float(), norm='backward')
    sty_fft = torch.fft.rfft2(style.float(), norm='backward')
    src_amplitude, src_phase = torch.abs(src_fft), torch.angle(src_fft)
    if uncertainty is None:
        # Preserve the supplied global-branch behavior, independent of strength.
        coefficient = torch.ones_like(src_amplitude) * np.random.uniform(0.1, 0.5)
    else:
        coefficient = frequency_mask(uncertainty, source.shape, strength, spec)
    amplitude = (1 - coefficient) * src_amplitude + coefficient * torch.abs(sty_fft)
    phase = src_phase
    if not spec.phase_lock:
        difference = torch.angle(sty_fft) - src_phase
        difference = torch.atan2(torch.sin(difference), torch.cos(difference))
        phase = src_phase + coefficient * difference
    result = torch.fft.irfft2(amplitude * torch.exp(1j * phase), s=source.shape[-2:], norm='backward')
    return result.to(source.dtype)


def loss_step(model, criterion, batch, epoch, spec, cpu_offload=False, total_epochs=49):
    img, target, div_img, div_target = [x.cuda(non_blocking=True) for x in batch]
    target, div_target = target.long(), div_target.long()
    strength, weight = strengths(epoch, spec, total_epochs)
    branch = 'disabled'
    offload = torch.autograd.graph.save_on_cpu(pin_memory=True) if cpu_offload else contextlib.nullcontext()
    with offload, torch.autocast('cuda', dtype=torch.float16):
        if spec.baseline:
            paired_img = torch.cat((img, div_img), dim=0)
            paired_target = torch.cat((target, div_target), dim=0)
            score, logit, features = model(paired_img)
            clean = criterion(logit, score, paired_target).mean()
            logit_loss = feature_loss = torch.zeros((), device=img.device)
            branch = 'paired_generated'
        else:
            score, logit, features = model(img)
            clean = criterion(logits=logit, anomaly_score=score, targets=target,
                              freq_features=features, input_imgs=img).mean()
            logit_loss = feature_loss = torch.zeros((), device=img.device)
            if strength > 0.01:
                with torch.no_grad():
                    p = random.random()
                    branch = 'uncertainty' if p < 0.4 else 'global' if p < 0.7 else 'identity'
                    if branch == 'uncertainty':
                        u = score.detach().unsqueeze(1)
                        flat = u.view(u.shape[0], -1)
                        lo = flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
                        hi = flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
                        mask = (u - lo) / (hi - lo + 1e-6)
                        aug = mix_variant(img, div_img, mask, strength, spec)
                    elif branch == 'global':
                        aug = mix_variant(img, div_img, None, strength * 0.8, spec)
                    else:
                        aug = img.clone()
                    aug = torch.clamp(aug, img.min(), img.max())
                _, logit_aug, features_aug = model(aug)
                if spec.consistency:
                    logit_loss = F.kl_div(F.log_softmax(logit_aug, dim=1), F.softmax(logit.detach(), dim=1), reduction='mean')
                    if spec.fem:
                        feature_loss = F.mse_loss(features_aug, features.detach(), reduction='mean')
        total = clean + (spec.lambda_logit * logit_loss + spec.lambda_feature * feature_loss) * weight
    info = dict(loss=float(total.detach()), loss_clean=float(clean.detach()),
                loss_logit=float(logit_loss.detach()), loss_feature=float(feature_loss.detach()),
                strength=strength, consistency_weight=weight, branch=branch, variant=spec.name)
    if not all(np.isfinite(info[k]) for k in ('loss', 'loss_clean', 'loss_logit', 'loss_feature')):
        raise FloatingPointError(f'Nonfinite loss in {spec.name}: {info}')
    return total, info
