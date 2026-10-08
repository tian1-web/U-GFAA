# Third-party source notices

The supplied U-GFAA source is based on [MultiShiftSeg](https://github.com/gaozhitong/MultiShiftSeg), by Zhitong Gao, Bingnan Li, Mathieu Salzmann, and Xuming He (NeurIPS 2024). The recorded upstream revision is 2f7404545c5984de2c2f85b652f23eb87d00401f. The original Apache-2.0 license is preserved in the repository root and this directory. The historical upstream README is retained under source_snapshot; it describes MultiShiftSeg, not a separate U-GFAA implementation.

The source snapshot already contains U-GFAA-specific modifications. Further public-release modifications are limited to path configuration, removal of private machine metadata, and refreshed manifest hashes, as recorded in release_provenance.json. Existing license headers are retained.

| Included component | Source and license evidence |
| --- | --- |
| deepv3/deepv3.py and upstream counterpart | Thalles Santos Silva / sthalles/deeplab_v3, MIT header |
| deepv3/Resnet.py | PyTorch torchvision, BSD-3-Clause header |
| deepv3/SEresnext.py | Remi Cadene / pretrained-models.pytorch, BSD-3-Clause header |
| deepv3/wider_resnet.py | Mapillary / inplace_abn, BSD-3-Clause header |
| mask2former | Meta / facebookresearch/Mask2Former, MIT; retained for snapshot completeness and not required by the documented CNN commands |
| mask2former DETR-derived files | facebookresearch/detr, Apache-2.0 |
| mask2former deformable-attention ops | SenseTime / fundamentalvision/Deformable-DETR, Apache-2.0 source headers |
| Swin backbone | Microsoft MIT source header; the referenced Swin semantic-segmentation repository also carries the MMSegmentation Apache-2.0 license |
| Cityscapes label definitions | mcordts/cityscapesScripts, MIT |
| Anomaly Mix helper functions and semantic mapper | Source comments acknowledge tianyu0207/PEBAL. A standalone PEBAL license was not identified at its public repository root as of 2026-10-08; no new license is asserted for that upstream contribution. The existing attribution is retained. |

These notices do not replace the component-specific licenses. Dataset and pretrained-checkpoint rights are separate; neither data nor weights are included.
