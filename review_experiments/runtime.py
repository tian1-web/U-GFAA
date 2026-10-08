"""Local execution support for the pinned, user-supplied training implementation.

Public-release modification (2026-10-08): configurable data/checkpoint paths only.

The default source_ddp mode preserves the supplied forward, loss, augmentation,
and epoch schedule. Windows uses one CUDA device and ordinary BatchNorm.
"""
from pathlib import Path
import contextlib
import hashlib
import importlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

EXP = Path(__file__).resolve().parent
ROOT = EXP.parent
SOURCE = EXP / 'source_snapshot'
DATA = Path(os.environ.get('U_GFAA_DATA_ROOT', ROOT / 'data')).expanduser().resolve()
WEIGHTS = Path(os.environ.get('U_GFAA_WEIGHTS', ROOT / 'pretrained_weights/cityscapes_best.pth')).expanduser().resolve()
sys.path.insert(0, str(SOURCE))


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bootstrap(run_id):
    """Confine the legacy parser's config side effects to this experiment folder."""
    os.chdir(EXP)
    previous = sys.argv
    sys.argv = ['source_runtime', '--cfg', str(SOURCE / 'exps/DeepLab.yaml'), '--id', run_id]
    try:
        from lib.configs.parse_arg import opt
        from lib.network.deepv3 import DeepWV3Plus
    finally:
        sys.argv = previous
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    return opt, DeepWV3Plus


def seed_all(seed):
    from lib.utils import random_init
    random_init(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def make_dataset(opt, selection_path=None):
    from lib.dataset.cityscapes import DiverseCityscapes
    from lib.utils.img_utils import Compose, ToTensor, RandCrop, Normalize
    transform = Compose([ToTensor(), RandCrop(tuple(opt.data.crop_size)),
                         Normalize(mean=opt.data.mean, std=opt.data.std)])
    from autodl_portability import replay_enumeration
    with replay_enumeration(DATA, EXP / 'dataset_enumeration.json'):
        ds = DiverseCityscapes(root=str(DATA / 'cityscapes'),
                              generation_root=str(DATA / 'DTWP_ADE_final'),
                              coco_root=str(DATA / 'coco'), split='train', transform=transform,
                              anomaly_mix=opt.data.anomaly_mix, mixup=opt.data.mixup, freq_aug=False)
    if len(ds) != 2917 or len(ds.coco_targets) != 19058:
        raise RuntimeError(f'Unexpected dataset counts: {len(ds)}, {len(ds.coco_targets)}')
    fields = ('images', 'targets', 'generated_images', 'generated_targets', 'coco_images', 'coco_targets')
    for name in fields:
        missing = [p for p in getattr(ds, name) if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f'{name}: {len(missing)} missing; first {missing[0]}')
    selection = {name: getattr(ds, name) for name in fields}
    if selection_path:
        selection_path = Path(selection_path)
        if selection_path.exists():
            old = json.loads(selection_path.read_text(encoding='utf-8'))
            if old != selection:
                raise RuntimeError('Seeded dataset selection differs from saved run manifest')
        else:
            atomic_json(selection_path, selection)
    return ds


def make_model(model_cls, weight_path=WEIGHTS):
    model = model_cls(19)
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    state = checkpoint.get('state_dict', checkpoint)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    missing, extra = model.load_state_dict(state, strict=False)
    allowed = ('freq_enhancement.', 'ood_head.')
    unexpected_missing = [k for k in missing if not k.startswith(allowed)]
    unexpected_extra = [k for k in extra if k != 'criterion.nll_loss.weight']
    if unexpected_missing or unexpected_extra:
        raise RuntimeError(f'Unexpected initial weight mismatch: {unexpected_missing}; {extra}')
    model.uncertainty_func_init()
    meta = {'path': str(weight_path), 'sha256': sha256(weight_path),
            'missing_new_module_keys': missing, 'unused_pretraining_criterion_keys': extra,
            'source_checkpoint_keys': list(checkpoint)[:20],
            'parameters': sum(p.numel() for p in model.parameters())}
    del checkpoint, state
    return model.cuda(), meta


def configure(model, opt, stage):
    names = list(opt.model.trainable_params_name if stage == 1 else opt.model.trainable_params_name_update)
    if 'freq_enhancement' not in names:
        names.append('freq_enhancement')
    params = []
    for name, p in model.named_parameters():
        p.requires_grad_(any(s in name for s in names))
        if p.requires_grad:
            params.append(p)
    lr = opt.train.lr if stage == 1 else opt.train.lr_update
    return torch.optim.Adam(params, lr=lr, weight_decay=opt.train.weight_decay)


def curriculum_strength(epoch, mode='source'):
    if mode == 'fixed':
        return 0.6
    if mode == 'off':
        return 0.0
    if mode != 'source':
        raise ValueError(mode)
    return 0.6 if epoch < 35 else max(0.0, 0.6 * (1 - (epoch - 35) / 15))


def loss_step(model, criterion, batch, epoch, curriculum='source', force_branch=None,
              cpu_offload=False):
    """Default operations follow train_deeplab_ddp.py, including its clean-only RCL input."""
    from lib.utils.freq_aug import uncertainty_aware_amp_mix
    img, target, div_img, div_target = [x.cuda(non_blocking=True) for x in batch]
    target = target.long()
    strength = curriculum_strength(epoch, curriculum)
    branch = 'disabled'
    offload = torch.autograd.graph.save_on_cpu(pin_memory=True) if cpu_offload else contextlib.nullcontext()
    with offload, torch.autocast('cuda', dtype=torch.float16):
        score, logit, features = model(img)
        loss_clean = criterion(logits=logit, anomaly_score=score, targets=target,
                               freq_features=features, input_imgs=img).mean()
        loss_logit = loss_feat = torch.zeros((), device=img.device)
        if strength > 0.01:
            with torch.no_grad():
                p = random.random()
                branch = force_branch or ('uncertainty' if p < 0.4 else 'global' if p < 0.7 else 'identity')
                if branch == 'uncertainty':
                    u = score.detach().unsqueeze(1)
                    flat = u.view(u.shape[0], -1)
                    lo = flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
                    hi = flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
                    mask = (u - lo) / (hi - lo + 1e-6)
                    aug = uncertainty_aware_amp_mix(img, div_img, mask, strength=strength)
                elif branch == 'global':
                    aug = uncertainty_aware_amp_mix(img, div_img, None, strength=strength * 0.8)
                elif branch == 'identity':
                    aug = img.clone()
                else:
                    raise ValueError(branch)
                aug = torch.clamp(aug, img.min(), img.max())
            _, logit_aug, features_aug = model(aug)
            loss_logit = F.kl_div(F.log_softmax(logit_aug, dim=1),
                                 F.softmax(logit.detach(), dim=1), reduction='mean')
            loss_feat = F.mse_loss(features_aug, features.detach(), reduction='mean')
        total = loss_clean + (loss_logit + 0.1 * loss_feat) * (strength / 0.6)
    info = {'loss': float(total.detach()), 'loss_clean': float(loss_clean.detach()),
            'loss_logit': float(loss_logit.detach()), 'loss_feature': float(loss_feat.detach()),
            'strength': strength, 'branch': branch}
    if not all(np.isfinite(info[k]) for k in ('loss', 'loss_clean', 'loss_logit', 'loss_feature')):
        raise FloatingPointError(f'Nonfinite source loss: {info}')
    return total, info


def compact_state(model, opt):
    """Store every mutable tensor; frozen parameters are reconstructed from hashed initial weights."""
    prefixes = tuple(opt.model.trainable_params_name_update) + ('freq_enhancement',)
    mutable = {name for name, _ in model.named_buffers()}
    mutable.update(name for name, _ in model.named_parameters() if any(s in name for s in prefixes))
    state = model.state_dict()
    for name, p in model.named_parameters():
        if p.requires_grad and name not in mutable:
            raise RuntimeError(f'Checkpoint would omit trainable parameter: {name}')
    return {name: state[name].detach().cpu().clone() for name in sorted(mutable)}


def load_compact(model, state, opt):
    expected = set(compact_state(model, opt))
    if set(state) != expected:
        raise RuntimeError('Checkpoint mutable tensor inventory differs from current model')
    model.load_state_dict(state, strict=False)


def make_eval_datasets(opt, include_fishyscapes=False):
    from lib.dataset.anomaly import RoadAnomaly, RoadAnomaly21, RoadObstacle21
    from lib.utils.img_utils import Compose, ToTensor, Normalize
    tf = Compose([ToTensor(), Normalize(mean=opt.data.mean, std=opt.data.std)])
    datasets = {
        'RoadAnomaly': RoadAnomaly(root=str(DATA / 'road_anomaly'), transform=tf),
        'RoadAnomaly21_validation': RoadAnomaly21(root=str(DATA / 'segment_me/dataset_AnomalyTrack'), transform=tf),
        'RoadObstacle21_validation': RoadObstacle21(root=str(DATA / 'segment_me/dataset_ObstacleTrack'), transform=tf)}
    expected = [60, 10, 30]
    if include_fishyscapes:
        from lib.dataset.fishyscapes import Fishyscapes
        for split in ('Static', 'LostAndFound'):
            datasets[f'Fishyscapes_{split}'] = Fishyscapes(root=str(DATA / 'fishyscapes'), split=split, transform=tf)
        expected += [30, 100]
    if [len(d) for d in datasets.values()] != expected:
        raise RuntimeError('Evaluation dataset counts changed')
    return datasets
