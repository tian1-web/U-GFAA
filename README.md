# U-GFAA

Code accompanying **Uncertainty-Guided Frequency Adversarial Adaptation Method for Out-of-Distribution Detection in Autonomous Driving**.

This repository contains the supplied DeepLabV3+ implementation and the training/evaluation programs used for the supplementary 49-epoch experiments. It builds on [MultiShiftSeg](https://github.com/gaozhitong/MultiShiftSeg), with uncertainty-guided frequency augmentation, an auxiliary frequency branch, consistency losses, and controlled ablations.

The release preserves the behavior of the supplied implementation. Its supplementary training protocol and auxiliary FEM connection differ from the manuscript description. See [Implementation and experiment scope](docs/IMPLEMENTATION.md) before interpreting results. This release does not claim to reproduce every number in the manuscript tables.

## Contents

- `review_experiments/source_snapshot/`: model, losses, frequency augmentation, datasets, metrics, and original training/configuration files.
- `review_experiments/train_multiseed49.py`: complete source implementation with fixed training pairs across seeds.
- `review_experiments/train_multiseed49_baseline.py`: adapted MultiShiftSeg comparison.
- `review_experiments/train_variants.py`: controlled component, mask, phase, and curriculum ablations.
- `review_experiments/evaluate_run.py`: reload and evaluate a completed run at its fixed final epoch.
- `review_experiments/upstream_snapshot/`: upstream files used for the baseline.
- `release_provenance.json`: release preparation and file hashes.

Datasets, pretrained/final model weights, private review correspondence, cloud access files, and environment binaries are not distributed here.

## Environment

The supplementary runs used Python 3.10.8, PyTorch 2.10.0 with CUDA 12.8, and torchvision 0.25.0 on RTX 4090 GPUs. Each run used one GPU. Install in a separate environment on a CUDA-capable machine:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

The historical `source_snapshot/environment.yml` is retained as source provenance; it is not the recommended environment for these supplementary commands. Transformer-related files in that snapshot are not used by the documented CNN workflow.

## Data and initialization

Prepare the datasets separately, subject to their original terms. The [upstream MultiShiftSeg instructions](https://github.com/gaozhitong/MultiShiftSeg#data-preparation) describe the training data and pretrained closed-world DeepLabV3+/WideResNet38 checkpoint.

The wrapper expects the following layout:

```text
data/
  cityscapes/                    # leftImg8bit and gtFine
  DTWP_ADE_final/                 # generated Cityscapes pairs
  coco/                          # COCO images and anomaly masks
  road_anomaly/                  # original and labels
  segment_me/
    dataset_AnomalyTrack/         # images and labels_masks
    dataset_ObstacleTrack/        # images and labels_masks
  fishyscapes/
    Static/                      # original and labels
    LostAndFound/                # original and labels
pretrained_weights/
  cityscapes_best.pth
```

Use absolute paths when your files are stored elsewhere:

```bash
export U_GFAA_DATA_ROOT=/path/to/data
export U_GFAA_WEIGHTS=/path/to/cityscapes_best.pth
```

The recorded input selections require 2,917 clean/generated training pairs and 19,058 COCO mask candidates. The release includes relative file lists, not the images. A different dataset version or missing generated pairs will fail the input checks rather than silently change the experiment. Evaluation expects RoadAnomaly (60 images), RoadAnomaly21 validation (10), RoadObstacle21 validation (30), Fishyscapes Static (30), and Fishyscapes LostAndFound (100). Fishyscapes labels must be encoded as 0=inlier, 1=anomaly, 255=ignore.


## Attribution and licensing

The implementation derives from [MultiShiftSeg](https://github.com/gaozhitong/MultiShiftSeg), associated with Gao et al., *Generalize or Detect? Towards Robust Semantic Segmentation Under Multiple Distribution Shifts*, NeurIPS 2024. The upstream commit recorded by the source snapshot is `2f7404545c5984de2c2f85b652f23eb87d00401f`.

The upstream Apache-2.0 license is included as `LICENSE`. Third-party notices and file-level MIT/BSD and other license headers are retained and continue to apply. Please acknowledge both U-GFAA and the upstream methods when using their respective contributions.
