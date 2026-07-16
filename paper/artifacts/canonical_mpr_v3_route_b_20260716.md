# canonical-mpr-v3 / Route B implementation record

The formal method now stores one compact canonical C-RADIO primitive field per
scene. DINOv3 and SAM3 remain frozen official pointwise views. Text descriptors
are rebuilt from the same field by a single global 3-D surface-region readout
that predicts a 1280-D RADIO backbone summary token, followed by the frozen
official `siglip2-g` summary head. Scene-specific crop-summary MPR is retained
only as an oracle.

## What changed

- Replaced the 2-D square-token bridge input with depth/pose-fused canonical
  ScanNet surfels, relative 3-D position, primitive scale, opacity/reliability,
  and explicit physical-scale conditioning.
- Regions are sampled and read on a DINO/SAM/geometry-conditioned surface graph
  at physical part/object/context scales; inference no longer defines regions
  by fixed-cardinality Euclidean kNN.
- Training uses official crop summary tokens from multiple real views. The
  token target is the medoid selected in official SigLIP2 descriptor space;
  descriptor loss covers every visible view and includes relation loss.
- The bridge split is scene-disjoint: 24 official ScanNet-train scenes versus
  8 official ScanNet-val scenes. No labels, instances, masks, or text are read.
- Formal cache generation decodes RADIO only from the canonical field. The old
  `radio_source=mpr` route is not present in the v3 builder.
- Default primitive text scoring is independent cosine, with a regression test
  proving query-set invariance. `scene_softmax` is a named compatibility variant.
- Primitive confidence now uses a geometric mean across conjunctive reliability
  channels instead of `amax`.

## Verification

Scene-disjoint validation improved the descriptor selection score from 0.7613
for the untrained residual baseline to 0.9265. All-view descriptor cosine is
0.9168 and summary-token cosine is 0.6834. Per physical scale, all-view cosine
is 0.9254 / 0.9164 / 0.9059 at 0.25 / 0.45 / 0.70 m.

Under the exact prior LERF compatibility protocol (`scene_softmax`, fixed 0.6,
no refinement), scene-macro mIoU rises from 0.2480 for the official-summary
sidecar oracle to 0.3433 for v3. Per scene: figurines 0.3067, ramen 0.4230,
teatime 0.3996, waldo-kitchen 0.2440. Waldo mIoU is nearly unchanged, while
localization rises from 0.4091 to 0.6364.

On the three frozen ScanNet scenes, calibration-free 19/15/10-class macro mIoU
is 0.4629 / 0.4753 / 0.5845, versus 0.2930 / 0.2957 / 0.3969 for the sidecar.
Corresponding macro mAcc is 0.6779 / 0.6911 / 0.7875.

The one-field 90%-of-sidecar acceptance gate is therefore passed on both
benchmarks. Derived semantic caches record the core and global readout sources
and can be deleted and rebuilt; they are not part of cold scene storage.

## ScanNet and pose-free image exemplar

`scannet_frames_25k` is suitable for training the global readout: it contains
1513 scenes with RGB, depth, pose, and separate color/depth intrinsics. It is
not sufficient to claim a strict pose-free result using the already trained
fields, because candidate query frames participated in geometry/MPR/render
fitting, and this directory does not contain the official 3-D aggregation and
segment-instance files needed for exact-instance metrics. Pose-free evaluation
is therefore deliberately not reported. A future run must freeze query frames
before geometry training and add official 3-D instance annotations.

## Remaining limitations

- Raw summary-token cosine (0.6834) still trails descriptor cosine; the official
  semantic space is reconstructed much better than the exact backbone token.
- The 0.70 m context scale is slightly weaker than the smaller scales.
- Warm caches are large (about 2.3--9.4 GiB per LERF scene); cold storage remains
  only 87--351 MiB plus a shared 1.41 MiB readout. Streaming/on-demand region
  readout should replace full warm materialization in the final system.
- LERF numbers above are explicitly the paper-compatible scene-softmax variant;
  a separate query-invariant scorer table should be added before paper freeze.
