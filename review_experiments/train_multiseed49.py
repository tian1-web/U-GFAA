"""Complete baseline and ablation training with explicit intervention records."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from runtime import (EXP, SOURCE, atomic_json, bootstrap, compact_state, configure,
                     load_compact, make_dataset, make_eval_datasets,
                     make_model, restore_rng, rng_state, seed_all, sha256)
from ablation_variants import (VARIANTS, adjust_config, model_class, finalize_model,
                               make_criterion, loss_step)
from training_budget import budget_note, budget_record, validate_epochs
from fixed_seed_training_data import make_dataset, policy_record


def atomic_torch_save(path, obj):
    temp = path.with_suffix('.tmp')
    torch.save(obj, temp)
    temp.replace(path)


def write_event(run, event):
    event = {'time': time.time(), **event}
    with (run / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(event, ensure_ascii=False) + '\n')


def evaluate(model, opt, run, epoch, best):
    from lib.utils.metric import eval_ood_measure
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    metrics = {}
    for name, ds in make_eval_datasets(opt).items():
        atomic_json(run / 'status.json', {'status': 'running', 'phase': 'evaluation',
                    'pid': os.getpid(), 'epoch': epoch, 'dataset': name, 'updated_at': time.time()})
        print(f'Epoch {epoch}: evaluating {name}, {len(ds)} images at original resolution', flush=True)
        predictions, labels = [], []
        with torch.no_grad():
            for data in DataLoader(ds, batch_size=1, num_workers=0, shuffle=False):
                image, target = data[0].cuda(), data[1]
                score, logit, features = model(image)
                if not torch.isfinite(score).all():
                    raise FloatingPointError(f'Nonfinite prediction on {name}: {data[2]}')
                predictions.append(score.cpu().numpy().reshape(-1))
                labels.append(target.numpy().astype(np.uint8).reshape(-1))
                del image, score, logit, features
        pred = np.concatenate(predictions)
        truth = np.concatenate(labels)
        del predictions, labels
        values = eval_ood_measure(pred, truth)
        if values is None or not np.isfinite(values).all():
            raise RuntimeError(f'Invalid evaluation result for {name}')
        metrics[name] = dict(zip(('AUROC', 'AUPRC', 'FPR_TPR95'), map(float, values)))
        metrics[name].update(images=len(ds), valid_pixels=int(np.count_nonzero(truth != 255)),
                             anomalous_pixels=int(np.count_nonzero(truth == 1)))
        print(f'Epoch {epoch}: {name}: {metrics[name]}', flush=True)
        write_event(run, {'kind': 'evaluation', 'epoch': epoch, 'dataset': name, **metrics[name]})
        if metrics[name]['AUPRC'] > best.get(name, {}).get('AUPRC', -1):
            best[name] = {'epoch': epoch, **metrics[name]}
            atomic_torch_save(run / f'{name}_best.pt',
                              {'mutable_state': compact_state(model, opt), 'epoch': epoch,
                               'metrics': metrics[name], 'selection': f'{name} AUPRC (validation-selected)'})
        del pred, truth
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(run / f'metrics_epoch_{epoch:02d}.json', metrics)
    atomic_json(run / 'best_validation_selected.json', best)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--variant', choices=sorted(VARIANTS), required=True)
    parser.add_argument('--cpu-offload', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--epochs', type=int, default=49)
    args = parser.parse_args()
    validate_epochs(args.epochs)
    spec = VARIANTS[args.variant]
    run_id = args.run_id or f'{spec.name}_seed{args.seed}'
    if Path(run_id).name != run_id:
        raise ValueError('run-id must be a plain directory name')
    run = EXP / 'runs' / run_id
    run.mkdir(parents=True, exist_ok=True)
    opt, cls = bootstrap(run_id)
    adjust_config(opt, spec)
    opt.train.n_epochs = args.epochs + 1
    seed_all(args.seed)
    ds = make_dataset(opt, run / 'dataset_selection.json')
    model, init_meta = make_model(model_class(cls, spec))
    model, init_meta = finalize_model(model, init_meta, spec)
    criterion = make_criterion(opt, spec)
    scaler = torch.amp.GradScaler('cuda')
    stage = 1
    optimizer = configure(model, opt, stage)
    # Exactly the supplied per-device batch and accumulation. World size is one.
    accumulation = 4
    batch_size = int(opt.train.train_batch)
    steps_per_epoch = len(ds) // batch_size
    config = {
        'run_id': run_id, 'seed': args.seed,
        'dataset_policy': policy_record(),
        'reference': 'upstream_snapshot/train_deeplab.py' if spec.baseline else 'source_snapshot/train_deeplab_ddp.py',
        'variant': spec.record(),
        'config': json.loads(json.dumps(opt)), 'initial_weights': init_meta,
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
        'world_size': 1, 'physical_batch': batch_size, 'accumulation': accumulation,
        'effective_batch': batch_size * accumulation, 'training_images': len(ds),
        'microbatches_per_epoch': steps_per_epoch, 'epochs': list(range(1, opt.train.n_epochs)),
        'curriculum': spec.schedule, 'cpu_offload': args.cpu_offload,
        'training_budget': budget_record(args.epochs),
        'changes_from_ddp_environment': [
            'Single CUDA GPU, ordinary BatchNorm (source SyncBatchNorm across two GPUs differs).',
            'Effective batch 8 (source two-GPU configuration has effective batch 16).',
            'Zero data-loader workers: source transformations retained, RNG sequence recorded for resume.',
            'Validation batch 1, original image resolution, full float32 model inference.',
            'Local dataset roots replace Linux paths; 2917 sources have supplied generated partners.',
            'Atomic compact checkpoints include all trainable tensors and all BN buffers; frozen weights reconstructed from hashed initialization.',
            'Each algorithmic intervention is recorded in variant. Remaining source behavior, including residual accumulated gradients across epoch boundaries, is retained.',
            'MultiShiftSeg uses paired clean/generated batches (four images per forward), upstream learning rates, AMP, and four-step gradient accumulation. Undefined empty-set contrastive means are zero, without skipping training batches.',
            budget_note(args.epochs)
        ],
        'program_sha256': {p.name: sha256(p) for p in (EXP / 'runtime.py', EXP / 'train_multiseed49.py', EXP / 'ablation_variants.py', EXP / 'training_budget.py', EXP / 'autodl_portability.py', EXP / 'dataset_enumeration.json', EXP / 'fixed_seed_training_data.py', EXP / 'multiseed49_fixed_selection.json')},
        'upstream_sha256': {p: sha256(EXP / 'upstream_snapshot' / p) for p in ('lib/loss.py', 'lib/network/deepv3/deepv3.py', 'exps/DeepLab.yaml')},
        'source_manifest_sha256': sha256(SOURCE / 'manifest.json'),
        'data_manifest_sha256': sha256(EXP / 'training_data_manifest.json')
    }
    config_path = run / 'run_config.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding='utf-8'))
        if previous != config:
            raise RuntimeError('Run configuration changed; refusing to silently resume a different experiment')
        if not args.resume:
            raise RuntimeError('Run already exists; use --resume')
    else:
        if args.resume:
            raise RuntimeError('Cannot resume an uninitialized run')
        atomic_json(config_path, config)
        for p in (EXP / 'runtime.py', EXP / 'train_multiseed49.py', EXP / 'ablation_variants.py', EXP / 'training_budget.py', EXP / 'autodl_portability.py', EXP / 'dataset_enumeration.json', EXP / 'fixed_seed_training_data.py', EXP / 'multiseed49_fixed_selection.json'):
            shutil.copy2(p, run / ('executed_' + p.name))
    epoch, next_batch, phase = 1, 0, 'train'
    microsteps = optimizer_steps = skipped_steps = 0
    best = {}
    optimizer.zero_grad(set_to_none=True)
    if args.resume:
        saved = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
        if saved['config_sha256'] != sha256(config_path):
            raise RuntimeError('Checkpoint/config identity mismatch')
        load_compact(model, saved['mutable_state'], opt)
        stage = saved['stage']
        optimizer = configure(model, opt, stage)
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        for name, p in model.named_parameters():
            if name in saved['gradients']:
                p.grad = saved['gradients'][name].to(p.device)
        epoch, next_batch, phase = saved['epoch'], saved['next_batch'], saved['phase']
        microsteps, optimizer_steps, skipped_steps = saved['microsteps'], saved['optimizer_steps'], saved['skipped_steps']
        best = saved['best']
        restore_rng(saved['rng'])
        del saved
    started = time.monotonic()

    def save_checkpoint():
        if shutil.disk_usage(run).free < 3 * 2**30:
            raise OSError('Less than 3 GiB free; refusing unsafe checkpoint overwrite')
        state = dict(mutable_state=compact_state(model, opt), optimizer=optimizer.state_dict(),
                     scaler=scaler.state_dict(), rng=rng_state(), epoch=epoch, next_batch=next_batch,
                     phase=phase, stage=stage, microsteps=microsteps, optimizer_steps=optimizer_steps,
                     skipped_steps=skipped_steps, best=best, config_sha256=sha256(config_path),
                     gradients={name: p.grad.detach().cpu() for name, p in model.named_parameters() if p.grad is not None})
        atomic_torch_save(run / 'last.pt', state)
        write_event(run, {'kind': 'checkpoint', 'epoch': epoch, 'next_batch': next_batch,
                          'phase': phase, 'optimizer_steps': optimizer_steps})

    if not args.resume:
        save_checkpoint()
    write_event(run, {'kind': 'start' if not args.resume else 'resume', 'pid': os.getpid(),
                      'epoch': epoch, 'next_batch': next_batch})
    print(f'Full independent training: {run_id} ({spec.name}); {len(ds)} samples, {steps_per_epoch} steps/epoch, epochs 1..{args.epochs}', flush=True)
    try:
        while epoch < opt.train.n_epochs:
            if phase == 'train':
                wanted_stage = 1 if epoch < opt.train.warmup_epoch else 2
                if stage != wanted_stage:
                    stage = wanted_stage
                    optimizer = configure(model, opt, stage)
                    optimizer.zero_grad(set_to_none=True)
                    write_event(run, {'kind': 'training_stage', 'stage': stage, 'epoch': epoch})
                sampler = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=args.seed)
                sampler.set_epoch(epoch)
                indices = list(sampler)[:steps_per_epoch * batch_size]
                batches = [indices[i:i + batch_size] for i in range(next_batch * batch_size, len(indices), batch_size)]
                loader = DataLoader(ds, batch_sampler=batches, num_workers=0, pin_memory=True)
                # A resumed iterator must not consume an extra global RNG draw.
                saved_rng = rng_state() if next_batch > 0 else None
                iterator = iter(loader)
                if saved_rng is not None:
                    restore_rng(saved_rng)
                model.train()
                for batch_idx, data in enumerate(iterator, start=next_batch):
                    tick = time.monotonic()
                    total, info = loss_step(model, criterion, data, epoch, spec,
                                            cpu_offload=args.cpu_offload, total_epochs=args.epochs)
                    scaler.scale(total / accumulation).backward()
                    microsteps += 1
                    if (batch_idx + 1) % accumulation == 0:
                        old_scale = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        if scaler.get_scale() < old_scale:
                            skipped_steps += 1
                        else:
                            optimizer_steps += 1
                        optimizer.zero_grad(set_to_none=True)
                    next_batch = batch_idx + 1
                    info.update(kind='train', epoch=epoch, batch=next_batch, microsteps=microsteps,
                                optimizer_steps=optimizer_steps, skipped_steps=skipped_steps,
                                scale=scaler.get_scale(), seconds=time.monotonic() - tick,
                                peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30)
                    write_event(run, info)
                    if next_batch <= 4 or next_batch % 20 == 0:
                        print(f'Epoch {epoch}/{args.epochs} batch {next_batch}/{steps_per_epoch}: loss={info["loss"]:.6f}; updates={optimizer_steps}; skipped={skipped_steps}', flush=True)
                        atomic_json(run / 'status.json', {'status': 'running', 'phase': 'train',
                                    'pid': os.getpid(), 'updated_at': time.time(), **info})
                    del total, data
                    if next_batch % 100 == 0:
                        save_checkpoint()
                phase = 'evaluation'
                save_checkpoint()
            if phase == 'evaluation':
                evaluate(model, opt, run, epoch, best)
                epoch += 1
                next_batch = 0
                phase = 'train'
                save_checkpoint()
        atomic_torch_save(run / 'final.pt', {'mutable_state': compact_state(model, opt),
                          'epoch': epoch - 1, 'config_sha256': sha256(config_path),
                          'selection': f'fixed final epoch {args.epochs}; not selected by validation'})
        atomic_json(run / 'status.json', {'status': 'complete', 'pid': os.getpid(),
                    'updated_at': time.time(), 'epochs_completed': epoch - 1,
                    'microsteps': microsteps, 'optimizer_steps': optimizer_steps, 'skipped_steps': skipped_steps})
        write_event(run, {'kind': 'complete', 'epochs_completed': epoch - 1,
                          'seconds_this_session': time.monotonic() - started})
    except BaseException as exc:
        atomic_json(run / 'status.json', {'status': 'failed', 'pid': os.getpid(),
                    'updated_at': time.time(), 'epoch': epoch, 'next_batch': next_batch,
                    'phase': phase, 'error': str(exc), 'traceback': traceback.format_exc(),
                    'resume_from': 'last.pt (last atomically completed checkpoint)'})
        write_event(run, {'kind': 'failed', 'epoch': epoch, 'next_batch': next_batch, 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
