# Implementation and experiment scope

This repository releases the available implementation and supplementary experiment programs without changing the training algorithm during packaging.

## FEM connection

In `review_experiments/source_snapshot/lib/network/deepv3/deepv3.py`, the frequency enhancement module produces auxiliary features. The ASPP prediction path continues to receive the original backbone features. The release therefore does not implement a FEM-to-ASPP replacement, and its component ablations must be interpreted as ablations of this auxiliary implementation.

## Training protocol

The documented supplementary commands use Adam, 49 epochs, 700 by 700 crops, a physical batch of two, and four-step gradient accumulation on a single GPU (effective batch eight for the source model). This differs from the manuscript's SGD/Poly, 80k-iteration, resized 768 by 768, distributed configuration. The baseline pairs clean and generated samples in its forward pass. Separate runs may be scheduled on separate GPUs; that is not distributed training of a single run.

The coupled source schedule is retained. The augmentation strength starts at 0.6 and decays after epoch 35; the consistency contribution is scaled with the augmentation strength. `curriculum_fixed` holds these at their initial values. The actual expressions and intervention definitions are authoritative in `runtime.py` and `ablation_variants.py`.

Without consistency, the source implementation's augmented forward can update BatchNorm running statistics without providing an augmented task-loss gradient. This matters when interpreting the incremental U-SAFA ablation.

## Baseline adaptation

The supplementary MultiShiftSeg comparison retains its paired clean/generated task loss and upstream learning rates, with mixed precision, gradient accumulation, and a guard for undefined empty-set contrastive means. Sixteen recorded generated masks use IDs 24/26 where the source anomaly encoding is 254. `baseline_label_adapter.py` maps these IDs only in the generated half of baseline batches, verifies the recorded input identities, and leaves the source PNG files unchanged. It is an adapted baseline, not an unmodified upstream benchmark run.

## Release provenance

Public packaging changes dataset/weight path configuration and removes personal machine paths. The public source manifest records the released bytes. New runs record those public hashes. Historical private run configurations are not rewritten to pretend that they were produced by the public package; strict reloading of a historical run may therefore reject a release-time hash mismatch.

Dataset manifests contain relative public dataset filenames and provenance information, not licensed image data. Reproducing a run requires the corresponding datasets and initial weights.

## Validation status

Release preparation checks Python syntax, manifest/file consistency, required input declarations, and exclusion of private materials. Packaging does not rerun training, benchmark hidden test sets, or establish that this public package reproduces every manuscript result. No inference of statistical significance should be made from code availability alone.
