# canonical-mpr-v3 evaluation closeout

This report is governed by
`canonical_mpr_v3_evaluation_freeze_20260716.yaml`.  All method outputs use one
query-independent compact canonical RADIO primitive field.  No row uses
benchmark masks/labels, query-set statistics, per-scene threshold tuning, or
test-set calibration before prediction.

## Text query

### LERF-OVS independent cosine

| scene | mIoU | LocAcc |
|---|---:|---:|
| Figurines | 0.279072 | 0.607143 |
| Ramen | 0.147029 | 0.563380 |
| Teatime | 0.313963 | 0.610169 |
| Waldo Kitchen | 0.137666 | 0.681818 |
| scene macro | 0.219432 | 0.615628 |
| sample micro (208 annotations) | 0.228940 | 0.600962 |

These are query-set-invariant raw normalized-cosine results.  The separately
audited `softmax_scene` paper-compatibility result is closed-taxonomy and must
not be merged with this row.

### ScanNet text category segmentation

The evaluation domain is the official ScanNet label mesh; primitive scores are
projected by fixed inverse-distance 8-NN, and classification is normalized
cosine argmax with the official SigLIP2-G text tower.  There is no logit
calibration, spatial postprocess, or class aliasing.

| class split | scene-macro mIoU | scene-macro mAcc |
|---:|---:|---:|
| 19 | 0.462859 | 0.677915 |
| 15 | 0.475278 | 0.691095 |
| 10 | 0.584505 | 0.787468 |

The three frozen validation scenes are `scene0062_00`, `scene0140_00`, and
`scene0200_00`; this is a controlled three-scene evaluation, not a full
ScanNet-val claim.

## Registered 2-D prompt

### NVOS strict unseen, eight scenes

Target RGB and target cameras are excluded from geometry, feature extraction,
and support.  Positive/negative registered scribbles compile to primitive
seeds in the official SAM3 capability space.  The graph solver is the frozen
signed seeded random-walker.

| readout stage | macro foreground IoU |
|---|---:|
| unary prior | 0.740130 |
| propagated | 0.749029 |
| connected (frozen final output) | 0.726404 |

The connected output has macro pixel accuracy 0.953906.  Signed propagation
provides a net gain, but the later fixed connected-component prior is not
universal: on Orchids it lowers IoU from 0.700091 to 0.482042.  Results were
not switched post hoc; all stages are disclosed.

### SPIn-NeRF nine-scene diagnostic

Running.  This is a full-reference-mask diagnostic on the nine complete
official RGB scenes; `fork` is unavailable.  It is not the same protocol as
SAGA's sparse-click benchmark and will not be presented as a same-protocol
main-table comparison.

## World-space 3-D point query

The frozen canonical core one-click result on the same three ScanNet scenes is
macro IoU 0.322015 (0.421774/0.271479/0.272792).  canonical-mpr-v3 adds only
the global text readout, so this canonical-mpr-v2 core result is unchanged.

## Verification

- focused canonical/readout/solver/text-query tests: 49/49 passed;
- paper build: `latexmk -pdf` passed;
- NVOS summary validates one protocol hash and four no-leakage safety fields
  for all eight scenes.
