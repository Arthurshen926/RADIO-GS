# canonical-mpr-v1 rebuild and task refresh — 2026-07-15

## Scope and correction

ScanNet `scene0000_00`, NVOS `fern`, and SPIn-NeRF `fern` were rebuilt from
raw RADIO, official DINOv3, and official SAM3 matched observations.  An audit
during the rebuild found that raster lifting previously recorded
`normalize_each_view=true` without executing it.  The corrected implementation
normalizes every 2D feature map before raster lifting and records both the
execution flag and stage.  A channel-chunked exact contribution accumulator was
added for million-primitive 4096-D DINO caches; it changes memory scheduling,
not the result.

All nine corrected caches have contract digest
`c6be3fe6ad1442f36be4a0918b8a390bd2894fc40985414c332ac6c3fb7ddd0b`.
Within each scene the three feature spaces have identical Gaussian rows,
selected views, valid rows, and responsibility-sidecar SHA-256.  No benchmark
mask, target RGB, or text query was opened during cache, field, capability-bank,
or graph construction.

| Scene | Views | Gaussians | Valid rows | Responsibility SHA-256 prefix |
|---|---:|---:|---:|---|
| ScanNet scene0000 | 120 | 81,369 | 65,102 | `37b3f2af8448` |
| NVOS fern | 19 | 760,715 | 297,200 | `71dc9a93d1f6` |
| SPIn fern | 20 | 1,314,717 | 771,292 | `395a96cf8403` |

## Canonical-field reconstruction

The architecture and optimization policy are unchanged: D256 shared
coefficient, D128 local branch, primitive fusion, and frozen official
capability losses.  ScanNet used 20 epochs; NVOS and SPIn met the fixed early
stop rule at epoch 5.

| Scene | raw RADIO mean cosine | DINO target mean cosine | SAM3 target mean cosine |
|---|---:|---:|---:|
| ScanNet scene0000 | 0.960487 | 0.976222 | 0.947372 |
| NVOS fern | 0.990282 | 0.992740 | 0.977635 |
| SPIn fern | 0.993521 | 0.994201 | 0.987876 |

## Protocol-preserving task refresh

Every comparison below keeps the old query, seed, prototype count, fixed score
threshold, graph solver, and metric implementation.  No test-set calibration
is used.  Only the observation cache, canonical field, derived official
DINO/SAM bank, and query-free k=16 support graph are replaced.

| Task | Stage | Historical field | canonical-mpr-v1 | Delta |
|---|---|---:|---:|---:|
| ScanNet 3D point | unary mIoU | 0.190846 | 0.183028 | -0.007819 |
| ScanNet 3D point | propagated mIoU | 0.194905 | 0.189502 | -0.005403 |
| ScanNet 3D point | connected/core mIoU | 0.303096 | 0.294615 | -0.008481 |
| ScanNet 3D point | Acc@0.25 | 0.476190 | 0.523810 | +0.047619 |
| ScanNet 3D point | Acc@0.50 | 0.174603 | 0.126984 | -0.047619 |
| NVOS scribble | unary foreground IoU | 0.741275 | 0.720596 | -0.020679 |
| NVOS scribble | propagated foreground IoU | 0.740632 | 0.720338 | -0.020294 |
| NVOS scribble | connected foreground IoU | 0.744900 | 0.730200 | -0.014700 |
| SPIn full reference mask | unary foreground IoU | 0.403194 | 0.503700 | +0.100506 |
| SPIn full reference mask | propagated foreground IoU | 0.933798 | 0.940747 | +0.006949 |
| SPIn full reference mask | connected foreground IoU | 0.942710 | 0.944605 | +0.001895 |

The NVOS loss appears before propagation and is therefore primarily a
prompt-to-primitive discrimination change, not an MPR propagation failure.
SPIn improves strongly at unary readout and slightly at the final mask.  The
ScanNet distribution shifts: more instances exceed IoU 0.25 while fewer exceed
0.50, with a small macro-IoU decrease.

### Optional ScanNet text variant

The raw/DINO/SAM rebuild does not itself define an official text-aligned
primitive descriptor.  For completeness, a separate learned-bridge variant
uses the frozen global region-summary bridge trained on 15,000 COCO crops with
image holdout, followed by the official SigLIP2 summary head and official text
tower.  It does not use the ScanNet vocabulary, labels, aliases, calibration,
or spatial postprocessing during cache construction or inference.

| Split | mIoU | mAcc |
|---|---:|---:|
| ScanNet-19 | 0.263652 | 0.491785 |
| ScanNet-15 | 0.266225 | 0.509902 |
| ScanNet-10 | 0.402296 | 0.676359 |

This is reported as a learned-bridge variant, not as direct official spatial
adaptor performance and not as evidence that the raw/DINO/SAM-only chain is
text aligned.

## Artifacts

All new caches, fields, capability banks, graphs, and evaluations are under
`output/optimization_20260715/canonical_v1_rebuild/`.  Focused lifting-contract,
capability-target, ScanNet-point, and registered-prompt tests pass:
**23 passed**.
