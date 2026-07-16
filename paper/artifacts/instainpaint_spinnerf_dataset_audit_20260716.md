# InstaInpaint `spinnerf_dataset` local audit (2026-07-16)

## Scope and conclusion

Audited path:
`/mnt/pool/sqy/3d_understanding/spinnerf_dataset`.

The payload is structurally complete for the InstaInpaint/SPIn-NeRF **object
removal and inpainting** split: ten scenes, each with 60 train and 40 validation
views.  It is not the ten-scene SPIn-NeRF multiview-segmentation benchmark used
by RADIO-GS registered-prompt evaluation, and it does not contain that
benchmark's `fork (nerf_supervision)` scene.

## Inventory and integrity

- Total: 2,195,282,336 bytes, 5,050 files.
- Scenes: `1`, `2`, `3`, `4`, `7`, `9`, `10`, `12`, `book`, `trash`.
- Per scene: 100 RGB PNGs, 100 camera `.ptz`, 100 instance `.ptz`, 100
  confidence `.ptz`, 100 instance-visualization PNGs, two scene/dataset metadata
  files, and three split lists.
- Every RGB is a readable 1008 x 567 RGB PNG; every instance visualization is a
  readable PNG.
- All 3,020 compressed tensor/metadata files decompress and deserialize.
  Camera records contain rotation, translation, camera type, intrinsics,
  distortion, and image size; camera numeric fields are finite. Instance and
  confidence tensors are 567 x 1008.
- In every scene the five modalities have identical frame stems. `train.txt`
  contains 60 unique frames and `valid.txt` contains 40 unique frames; they are
  disjoint and their union equals all 100 frames.

The scene semantics reported for this object-removal cohort are `1` (bench),
`2` (tree), `3` (backpack), `4` (stairs), `7` (well), `9` (wall), `10`
(yard), `12` (garden), `book`, and `trash`.  None is the multiview-segmentation
Fork capture.

## Prompt benchmark assets already used by RADIO-GS

All protocol assets live below
`/mnt/pool/sqy/3d_understanding/segmentation_benchmarks`.

- NVOS annotations:
  `NVOS/official_annotations/llff`
- NVOS source RGB and cameras:
  `NVOS/llff_undistorted`
- Frozen NVOS manifest:
  `manifests/nvos_strict_unseen_v1.json`
- Prepared eight-task NVOS jobs and predictions:
  `gaussfm_jobs/nvos_strict_unseen_v1`
- SPIn-NeRF multiview segmentation masks:
  `SPIn-NeRF/multiview_annotations`
- SPIn-NeRF source RGB/cameras for the available nine scenes:
  `SPIn-NeRF/source_images`
- Official downloaded archives:
  `SPIn-NeRF/official/SPIn-NeRF`

The strict SPIn-NeRF prompt cohort remains 9/10 RGB-complete.  Its annotation
folder has all 38 Fork masks, but `SPIn-NeRF/source_images/fork` has no verified
full RGB/camera bundle.  Annotation cutouts are not admissible substitutes
because they remove background pixels and disclose the evaluation mask.

Earlier current-method prompt diagnostics are retained under:

- `output/optimization_20260715/canonical_v1_rebuild/nvos_fern`
- `output/optimization_20260715/canonical_v1_rebuild/spin_fern`

The older full eight-task NVOS frozen evaluation is under
`segmentation_benchmarks/gaussfm_jobs/nvos_strict_unseen_v1/evaluation.json`.
