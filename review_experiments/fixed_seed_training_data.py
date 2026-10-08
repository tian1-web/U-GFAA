"""Hold audited training pairs fixed while varying the training random seed.

The legacy constructor still consumes its original random draws. Only its two
generated-pair lists are replaced before runtime checks file existence. For
seed 0, these lists are unchanged. No image, label, transform or loss is edited.
"""
from contextlib import contextmanager
import json
from pathlib import Path, PurePosixPath

import runtime

MANIFEST = runtime.EXP / 'multiseed49_fixed_selection.json'


def expected_selection():
    record = json.loads(MANIFEST.read_text(encoding='utf-8'))
    result = {}
    for name, paths in record['selection'].items():
        for path in paths:
            pure = PurePosixPath(path)
            if pure.is_absolute() or '..' in pure.parts or '\\' in path:
                raise RuntimeError('Unsafe fixed-selection relative path')
        result[name] = [str(runtime.DATA / PurePosixPath(path)) for path in paths]
    return result


def policy_record():
    return dict(name='fixed_seed0_training_pairs_v1', manifest=MANIFEST.name,
                manifest_sha256=runtime.sha256(MANIFEST),
                fixed='All clean/generated image pairs, labels and COCO candidate lists use the audited seed-0 selection.',
                varied='New-module initialization, online augmentation and loss randomness, and DistributedSampler seed equal the declared training seed.',
                constructor_rng='Original per-seed constructor draws are retained before replacing generated lists; replacement consumes no random draws.',
                seed0_equivalence='Seed 0 retains identical input lists, constructor RNG consumption and sampler order to the original ported trainer.',
                sampling_seed='args.seed; seed 0 equals the original default DistributedSampler seed')


@contextmanager
def fixed_pairs_context():
    import lib.dataset.cityscapes as dataset_module
    original = dataset_module.DiverseCityscapes
    expected = expected_selection()

    class FixedPairsDataset(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # These lists must already match. Only the generated pairing varies
            # during the original constructor's random selection.
            for name in ('images', 'targets', 'coco_images', 'coco_targets'):
                if getattr(self, name) != expected[name]:
                    raise RuntimeError(f'Fixed-input source ordering changed: {name}')
            for name in ('generated_images', 'generated_targets'):
                if len(getattr(self, name)) != len(expected[name]):
                    raise RuntimeError(f'Fixed-input candidate count changed: {name}')
                setattr(self, name, list(expected[name]))

    dataset_module.DiverseCityscapes = FixedPairsDataset
    try:
        yield
    finally:
        dataset_module.DiverseCityscapes = original


def make_dataset(opt, selection_path=None):
    with fixed_pairs_context():
        return runtime.make_dataset(opt, selection_path)
