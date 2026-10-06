# External implementation references

`radio_gs/models/pretrained_mask_adapter.py` adapts the inference operations of
[hustvl/MaskAdapter](https://github.com/hustvl/MaskAdapter), commit
`c0516d8a548d90055c3dca7f2a9b4281a4da842f`, including the MaskAdapter head,
ConvNeXt block and FC-CLIP feature pooling. The upstream work is distributed
under [Apache-2.0](docs/licenses/MaskAdapter-Apache-2.0.txt). FC-CLIP backbone
code carries Copyright (2023) Bytedance Ltd. and/or its affiliates; ConvNeXt
blocks trace to the Facebook Research ConvNeXt implementation.

The pretrained checkpoint is the authors' `owl10/Mask-Adapter` release,
`fcclip_convnext_large_maskadapter.pth`, revision
`99662f6a64430bdb458f56f8285bd13e2bd1ac06`. It is an external source teacher.
Its architecture, training and pretrained capability are not RADIO-GS contributions.

`radio_gs/models/pretrained_panoptic_scope.py` also invokes the unchanged local
upstream full model and its COCO panoptic inference configuration to compile
an offline entity-scope prior. It loads the same full released checkpoint;
the omitted `criterion.empty_weight` tensor is a training-loss weight only.
This prior's masks and pretrained recognition are external teacher outputs,
not a newly trained RADIO-GS segmentation model.

`radio_gs/querying/witnessed_observation_scope.py` studies third-view containment
inspired by [MaskClustering](https://arxiv.org/html/2401.07745v2). It is an
independent implementation on overlapping SAM proposals, with generating views
excluded and seed neighborhoods retained, not a reproduction of that work's
panoptic predictor or iterative clustering algorithm.
