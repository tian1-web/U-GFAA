"""Repair two audited generated-anomaly encodings for the paired baseline only."""
import json
from pathlib import PurePosixPath

import torch
from torch import nn

import runtime
from ablation_variants import make_criterion as original_make_criterion

AUDIT = runtime.EXP / 'multiseed49_baseline_label_audit.json'


def label_policy_record():
    audit = json.loads(AUDIT.read_text(encoding='utf-8'))
    return dict(name='audited_generated_anomaly_24_26_to_254_v1',
                adapter_sha256=runtime.sha256(runtime.EXP / 'baseline_label_adapter.py'),
                audit_file=AUDIT.name, audit_sha256=runtime.sha256(AUDIT),
                audited_files=len(audit['affected_files']),
                mapping={'24': 254, '26': 254},
                scope='Only the generated half of paired baseline targets; original PNG files and clean targets are unchanged.',
                rationale='Sixteen generated anomaly masks contain 24/26 instead of the source generator defined anomaly ID 254. Raw-file hashes and region comparisons are recorded in the audit.',
                valid_clean_ids='0..18, source OOD IDs 100..254, and ignore 255; all other values fail.',
                valid_generated_ids='0..18, source OOD IDs 100..254, and ignore 255 after mapping; all other values fail.',
                loss='Retain the previously audited upstream relative-contrastive loss and empty-set guard. Corrected pixels remain OOD and are not ignored.')


def verify_audited_inputs():
    audit = json.loads(AUDIT.read_text(encoding='utf-8'))
    if len(audit['affected_files']) != 16 or audit['mapping'] != {'24': 254, '26': 254}:
        raise RuntimeError('Unexpected baseline label audit')
    from fixed_seed_training_data import expected_selection
    selected = set(expected_selection()['generated_targets'])
    for row in audit['affected_files']:
        relative = PurePosixPath(row['relative_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Unsafe audited label path')
        path = runtime.DATA / relative
        if str(path) not in selected or not path.is_file() or runtime.sha256(path) != row['sha256']:
            raise RuntimeError(f'Audited label identity mismatch: {path}')
    return dict(status='passed', verified_files=16, audit_sha256=runtime.sha256(AUDIT))


def normalize_paired_targets(targets, n_classes=19):
    if n_classes != 19 or targets.ndim != 3 or targets.shape[0] % 2 or targets.shape[0] == 0:
        raise ValueError('Expected a paired N,H,W target batch for 19 semantic classes')
    if targets.dtype != torch.long:
        raise ValueError('Expected integer long target IDs')
    half = targets.shape[0] // 2
    clean, generated = targets[:half], targets[half:]
    clean_valid = ((clean >= 0) & (clean < 19)) | ((clean >= 100) & (clean <= 255))
    if not bool(clean_valid.all()):
        raise ValueError(f'Illegal clean target IDs: {torch.unique(clean[~clean_valid]).tolist()}')
    generated_valid = ((generated >= 0) & (generated < 19)) | (generated == 24) | (generated == 26) | ((generated >= 100) & (generated <= 255))
    if not bool(generated_valid.all()):
        raise ValueError(f'Unaudited generated target IDs: {torch.unique(generated[~generated_valid]).tolist()}')
    # Clone even valid targets because the upstream pixel-selection loss mutates
    # its input. Neither that behavior nor this mapping alters source mask files.
    fixed = targets.clone()
    fixed_generated = fixed[half:]
    fixed_generated[(generated == 24) | (generated == 26)] = 254
    return fixed


class BaselineLabelAdapter(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, logits, anomaly_score, targets):
        if logits.shape[0] != targets.shape[0]:
            raise ValueError('Paired target batch and logits disagree')
        fixed = normalize_paired_targets(targets, n_classes=logits.shape[1])
        return self.criterion(logits, anomaly_score, fixed)


def make_criterion(opt, spec, verify_inputs=True):
    if spec.name != 'multishiftseg' or not spec.baseline:
        raise ValueError('The label adapter is restricted to the paired MultiShiftSeg baseline')
    if verify_inputs:
        verify_audited_inputs()
    return BaselineLabelAdapter(original_make_criterion(opt, spec))
