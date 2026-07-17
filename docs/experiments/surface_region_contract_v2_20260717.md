# SurfaceRegionContractV2 closeout (2026-07-17)

This report records promotion decisions made without benchmark-specific heads,
test-set calibration, SAM/GrabCut mask refinement, or scene-wise query softmax.

## Correctness

- Train and inference share `SurfaceRegionContractV2`: symmetric typed support
  graph, physical Dijkstra radii 0.25/0.45/0.70 m, 1.2x context shell, and
  deterministic distance/node-index truncation to 256 tokens.
- A real figurines audit compared 1,000 anchors at all three scales against the
  SciPy single-anchor reference. All 3,000 ordered row lists, core masks, and
  distances were exact; minimum Jaccard was 1.0.
- The bounded sparse implementation replaces a dense `batch x scene_nodes`
  distance allocation. It preserves the contract exactly and is required for
  practical cold streaming on 200k--430k-node graphs.
- The readout is anchor-conditioned, includes anchor-relative geometry and
  explicit core/context flags, and uses the same scale/reliability semantics in
  ScanNet training and canonical-field inference.
- Half-pixel sampling, constant opacity, and seed-only unary contract errors are
  fixed and covered by regression tests.

## Global readout

- Training: 24 disjoint ScanNet train scenes, 576 sampled regions.
- Validation: 8 disjoint ScanNet scenes, 192 sampled regions.
- Clean h256 checkpoint validation: selection score 0.948742; official summary
  token cosine 0.862125; mean descriptor cosine 0.958030; all-view descriptor
  cosine 0.939455.
- Canonical-residual noise augmentation was rejected: on 30k common figurines
  primitives, the clean readout achieved scale cosines
  0.80771/0.81135/0.81395 versus 0.79902/0.80311/0.80791 for the augmented
  variant.

## LERF independent-cosine result

All entries use fixed threshold 0.6, polygon-argmax localization, no semantic
post-processing, and max-after-cosine across the three frozen region scales.

| Scene | v3 mIoU | v2 region mIoU | Delta | v3 LocAcc | v2 LocAcc |
|---|---:|---:|---:|---:|---:|
| figurines | 0.2791 | 0.3845 | +0.1054 | 0.6071 | 0.6786 |
| ramen | 0.1470 | 0.2838 | +0.1368 | 0.5634 | 0.5352 |
| teatime | 0.3140 | 0.3741 | +0.0602 | 0.6102 | 0.5424 |
| waldo_kitchen | 0.1377 | 0.2166 | +0.0790 | 0.6818 | 0.5455 |
| scene macro | 0.2194 | 0.3148 | +0.0953 | 0.6156 | 0.5754 |

Category-macro mIoU rises from 0.2007 to 0.2972. The semantic change passes
the predeclared promotion gate (4/4 scenes improve), while localization does
not: broad/context-scale maxima flatten peaks in three scenes. No localization
scale rule should be promoted until frozen on non-LERF development data.

On figurines, the official crop-summary sidecar under the same cosine scorer
obtains 0.1831 sample mIoU and 0.5714 LocAcc, versus 0.3845 and 0.6786 for the
one-field readout. This closes the previously confounded one-field comparison.

## ScanNet text preservation check

Official label-mesh vertices, inverse-distance k=8 primitive projection,
official SigLIP2 text embeddings, no aliases/calibration/post-processing:

| Split | previous 3-scene macro | v2 region 3-scene macro |
|---|---:|---:|
| 19 classes | 0.4629 | 0.4694 |
| 15 classes | 0.4753 | 0.4700 |
| 10 classes | 0.5845 | 0.5844 |

The readout therefore does not buy LERF mIoU through a systematic ScanNet
regression. This remains a controlled three-scene check, not a full ScanNet-val
benchmark claim.

## Relation hierarchy decision

- Fixed geometry/DINO/SAM maximum-spanning hierarchies reached only about
  0.358 macro one-click ancestor oracle IoU, below the 0.55--0.70 gate.
- A global monotonic five-feature calibrator trained from official query-free
  SAM3 automatic masks reached validation AUC 0.5587 and was rejected.
- An 8D per-primitive relation-private code was then tested with four training
  and four held-out RGB frames in each of four scenes. Held-out AUC changes over
  the fixed base were +0.0065, +0.0072, +0.0049, and -0.0015. It was also
  rejected.
- Inspection shows that binary same/different labels from automatic masks are
  inconsistent across views because part, object, and context masks legitimately
  induce different merge levels. More capacity does not fix that supervision
  mismatch. Future hierarchy work must supervise an ordered merge scale (or a
  laminar region relation), not collapse all automatic masks to one binary edge.

## Cold streaming

The formal query path can decode canonical RADIO, gather regions, run the
shared readout and official summary head, and immediately retain only Q scalar
cosines per primitive. With the readout/scoring batch fixed to 1,024,
figurines (168,791 global rows, 21 queries) takes 45 s:

- disposable 1536D semantic cache: 1,017,885,394 bytes;
- warm scalar unary: 9,286,358 bytes;
- cold-stream fp16 unary: 9,288,327 bytes;
- cold/warm scalar scores are bitwise identical (max error 0), and the final
  localization/mIoU dictionaries are exactly equal.

Thus compactness is correctly claimed for persistent scene representation;
the multi-gigabyte descriptor cache is optional and not required by inference.

## Promotion summary

Promote the contract-correct anchor/core/context readout, independent
multiscale cosine, sparse bounded Dijkstra, and cold streaming. Reject canonical
noise augmentation, the binary global relation calibrator, and the current 8D
relation-private code. Keep the hierarchical region field as the next research
target, but do not describe it as solved in the paper.
