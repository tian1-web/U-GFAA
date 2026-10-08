# Public source layout

The supported entry points use the CNN DeepLabV3+ / WideResNet38 path. Historical snapshot training scripts and Mask2Former source are retained for provenance only; they are not validated release entry points.

## Explicit runtime file inventory

Cloud-archived files: runtime.py, train_multiseed49.py, train_multiseed49_baseline.py, ablation_variants.py, baseline_label_adapter.py, training_budget.py.

Local companion files: train_variants.py, evaluate_run.py, fixed_seed_training_data.py, autodl_portability.py. The last file only replays directory enumeration during local dataset construction and has no SSH or cloud connection functionality.

Data records: dataset_enumeration.json, training_data_manifest.json, multiseed49_fixed_selection.json, multiseed49_baseline_label_audit.json. These are relative file lists/audit records; no image data are included.

Upstream baseline files: lib/loss.py, lib/network/deepv3/deepv3.py, exps/DeepLab.yaml, train_deeplab.py.

## Dataset contract

Set U_GFAA_DATA_ROOT before invoking Python, or put data in the repository data/ directory. Set U_GFAA_WEIGHTS to the pretrained cityscapes_best.pth file, or use pretrained_weights/cityscapes_best.pth.

Training expects cityscapes/, DTWP_ADE_final/, and coco/ beneath the data root. The archived constructor requires 2,917 clean/generated training pairs and 19,058 COCO candidates. The included relative enumeration and fixed-selection files must match the downloaded dataset files. Labels and image content are never reconstructed from these manifests.

Evaluation expects road_anomaly/ (60 images), segment_me/dataset_AnomalyTrack/ (10 validation images), and segment_me/dataset_ObstacleTrack/ (30 validation images). evaluate_run.py additionally evaluates fishyscapes/ Static (30) and LostAndFound (100), so those data are required for that full final-evaluation command. Evaluation labels use 0/1/255, with 255 ignored. These are public validation subsets; this code does not submit to hidden test servers.

Baseline-only label adaptation retains the original audit of 16 generated masks and maps 24/26 to 254 in memory. The source PNG files and model algorithms are unchanged.

## Provenance and scope

Public-release paths and manifest hashes differ from historical experiment archives. New runs record the public-release hashes. Historical checkpoints are not included and should not be mixed with new configurations to bypass provenance checks. Exact historical metric reproduction has not been rerun as part of packaging. The release provenance file identifies source hashes and each packaging change.

No cloud connection/migration/backup utilities, datasets, checkpoints, logs, reviewer correspondence, or Python environments are included.

## Browser upload edition

This edition contains only visible files. Fourteen hidden notebook backup files, each byte-identical to its corresponding normal Python file, are omitted. The source manifest checks the 88 retained files and does not require those backups. Hidden Git configuration files are also omitted; they are not training or evaluation dependencies. No model, loss, training, or evaluation algorithm is changed.
