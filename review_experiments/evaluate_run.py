"""Reload a completed independent run and evaluate its fixed final checkpoint."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from runtime import (EXP, atomic_json, bootstrap, configure, load_compact, make_eval_datasets,
                     make_model, seed_all, sha256)
from training_budget import configured_epochs


def verify_run_provenance(config):
    """Reject evaluation with training code or source files changed since the run."""
    expected_files = {EXP / name: expected for name, expected in config['program_sha256'].items()}
    expected_files[EXP / 'source_snapshot/manifest.json'] = config['source_manifest_sha256']
    expected_files[EXP / 'training_data_manifest.json'] = config['data_manifest_sha256']
    for name, expected in config.get('upstream_sha256', {}).items():
        expected_files[EXP / 'upstream_snapshot' / name] = expected
    for path, expected in expected_files.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f'Training/evaluation provenance mismatch: {path}')
    source = EXP / 'source_snapshot'
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['source_files_sha256'].items():
        path = source / name.replace('\\', '/')
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f'Pinned source was changed: {name}')
    return len(manifest['source_files_sha256'])


def load_run(run_id):
    run = EXP / 'runs' / run_id
    config = json.loads((run / 'run_config.json').read_text(encoding='utf-8'))
    status = json.loads((run / 'status.json').read_text(encoding='utf-8'))
    epochs = configured_epochs(config)
    if status.get('status') != 'complete' or status.get('epochs_completed') != epochs:
        raise RuntimeError(f'Only a completed {epochs}-epoch run can produce final evaluation results')
    verify_run_provenance(config)
    opt, cls = bootstrap('evaluate_' + run_id)
    spec = None
    if 'variant' in config:
        from ablation_variants import VARIANTS, adjust_config, finalize_model, model_class
        spec = VARIANTS[config['variant']['name']]
        if spec.record() != config['variant']:
            raise RuntimeError('Saved intervention definition differs from current implementation')
        adjust_config(opt, spec)
        cls = model_class(cls, spec)
    seed_all(config['seed'])
    model, meta = make_model(cls)
    if spec is not None:
        model, meta = finalize_model(model, meta, spec)
    if meta['sha256'] != config['initial_weights']['sha256']:
        raise RuntimeError('Initial frozen weights differ from training')
    final_path = run / 'final.pt'
    saved = torch.load(final_path, map_location='cpu', weights_only=False)
    if saved['config_sha256'] != sha256(run / 'run_config.json') or saved['epoch'] != epochs:
        raise RuntimeError('Final checkpoint identity mismatch')
    # Recreate the training freeze flags before checking the compact inventory.
    # A fresh model otherwise marks its reconstructed frozen backbone trainable.
    configure(model, opt, stage=1 if epochs < opt.train.warmup_epoch else 2)
    load_compact(model, saved['mutable_state'], opt)
    del saved
    model.eval()
    return run, config, opt, model, sha256(final_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--save-predictions', action='store_true')
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError('Invalid run-id')
    run, config, opt, model, checkpoint_hash = load_run(args.run_id)
    from lib.utils.metric import eval_ood_measure
    output = run / 'final_evaluation'
    output.mkdir(exist_ok=True)
    final_epoch = configured_epochs(config)
    model_record = {'run_id': args.run_id, 'selection': f'fixed final epoch {final_epoch}',
                    'training_budget': config.get('training_budget'),
                    'checkpoint_sha256': checkpoint_hash, 'seed': config['seed'],
                    'run_config_sha256': sha256(run / 'run_config.json'),
                    'source_manifest_sha256': config['source_manifest_sha256'],
                    'training_program_sha256': config['program_sha256'],
                    'data_manifest_sha256': config['data_manifest_sha256'],
                    'save_predictions': args.save_predictions,
                    'eval_program_sha256': sha256(Path(__file__)),
                    'metric_program_sha256': sha256(EXP / 'source_snapshot/lib/utils/metric.py')}
    results = {}
    expected_epoch_metrics = json.loads((run / f'metrics_epoch_{final_epoch:02d}.json').read_text(encoding='utf-8'))
    for name, ds in make_eval_datasets(opt, include_fishyscapes=True).items():
        result_path = output / (name + '.json')
        if result_path.is_file():
            saved = json.loads(result_path.read_text(encoding='utf-8'))
            if saved.get('model') == model_record:
                results[name] = saved['metrics']
                continue
        tick = time.monotonic()
        predictions, labels, per_image = [], [], []
        prediction_dir = output / 'predictions' / name
        if args.save_predictions:
            prediction_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for index, data in enumerate(DataLoader(ds, batch_size=1, num_workers=0, shuffle=False)):
                image = data[0].cuda()
                score, logit, features = model(image)
                if not torch.isfinite(score).all():
                    raise FloatingPointError(f'Nonfinite final prediction: {name}, {index}')
                score_np = score[0].cpu().numpy().astype(np.float32)
                label_np = data[1][0].numpy().astype(np.uint8)
                unexpected = set(np.unique(label_np).tolist()) - {0, 1, 255}
                if unexpected:
                    raise ValueError(f'Unexpected evaluation labels in {name}: {unexpected}')
                predictions.append(score_np.reshape(-1))
                labels.append(label_np.reshape(-1))
                item = dict(index=index, name=data[2][0], image_path=ds.images[index],
                            label_path=ds.targets[index], shape=list(score_np.shape),
                            valid_pixels=int(np.count_nonzero(label_np != 255)),
                            anomaly_pixels=int(np.count_nonzero(label_np == 1)))
                if args.save_predictions:
                    # Lossless float32 predictions plus exact label IDs; no quantization.
                    path = prediction_dir / f'{index:04d}.npz'
                    temp = path.with_suffix('.tmp')
                    with temp.open('wb') as stream:
                        np.savez_compressed(stream, score=score_np, label=label_np)
                    temp.replace(path)
                    item['prediction_file'] = str(path.relative_to(run))
                    item['prediction_sha256'] = sha256(path)
                per_image.append(item)
                del image, score, logit, features
        scores, truth = np.concatenate(predictions), np.concatenate(labels)
        del predictions, labels
        values = eval_ood_measure(scores, truth)
        if values is None or not np.isfinite(values).all():
            raise RuntimeError(f'Invalid final metrics for {name}')
        metrics = dict(zip(('AUROC', 'AUPRC', 'FPR_TPR95'), map(float, values)))
        metrics.update(images=len(ds), valid_pixels=int(np.count_nonzero(truth != 255)),
                       anomalous_pixels=int(np.count_nonzero(truth == 1)))
        # Reloading a compact checkpoint must reproduce the last training evaluation.
        if name in expected_epoch_metrics:
            for metric in ('AUROC', 'AUPRC', 'FPR_TPR95'):
                if abs(metrics[metric] - expected_epoch_metrics[name][metric]) > 1e-7:
                    raise RuntimeError(f'Reloaded final checkpoint changed {name} {metric}')
        results[name] = metrics
        atomic_json(result_path, {'model': model_record, 'metrics': metrics, 'per_image': per_image,
                                  'seconds': time.monotonic() - tick})
        print(f'{args.run_id}: {name}: {metrics}', flush=True)
        del scores, truth
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(output / 'summary.json', {'status': 'complete', 'model': model_record, 'metrics': results})


if __name__ == '__main__':
    main()
