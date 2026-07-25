# Specificity-preserving text scale readout — 2026-07-24

## Status

This cross-task readout passes its declared ScanNet and LERF gates and is
retained in the v4 promotion candidate. The compact canonical field, global
region summary readout, official SigLIP2 text encoder, text prompts,
thresholds, and benchmark evaluators are unchanged.

The only change is the query-set-invariant reduction over the three ordered
physical region scales. Let \(c_{i,s}(q)\) be the independent normalized cosine
for primitive \(i\), scale \(s\), and query \(q\). Instead of always taking the
numerical maximum, the candidate selects

\[
s_i^*(q)=\min\{s:c_{i,s}(q)\geq \max_t c_{i,t}(q)-0.02\}.
\]

The selected unary is \(c_{i,s_i^*}(q)\). Scales are ordered local to
contextual. A zero margin is exactly value-equivalent to historical max
aggregation. The default evaluator and cache compiler behavior remains max.

## Parameter freeze

The absolute cosine margin `0.02` was declared before opening any LERF result.
It was evaluated once on the three controlled ScanNet text development scenes.
No sweep over LERF labels, queries, masks, per-scene rules, or test-set
calibration is used.

## ScanNet development gate

Official label-mesh vertices, official SigLIP2 text embeddings, fixed inverse
distance \(k=8\) projection, no aliases, no logit calibration, and no spatial
post-processing:

| Split | max-after-cosine | specificity 0.02 | Delta |
|---|---:|---:|---:|
| 19 classes | 0.469351 | 0.470437 | +0.001087 |
| 15 classes | 0.469953 | 0.471775 | +0.001822 |
| 10 classes | 0.584760 | 0.589430 | +0.004670 |

The three-scene macro gate passes. Scene0062 changes by only
-0.000167/-0.000111/-0.000499 for the 19/15/10 splits, while scene0140 and
scene0200 supply the net gain. This is a small readout improvement, not evidence
of a new semantic field.

## LERF gate

All four specificity unary caches were rendered and evaluated with the
unchanged independent-cosine LERF protocol:

- fixed peak-relative threshold 0.6;
- `polygon_argmax` localization;
- exact official annotation frames and query strings;
- no semantic post-processing, RGB refinement, SAM/GrabCut, or test-set
  calibration.

| Scene | max mIoU | specificity mIoU | Delta | max LocAcc | specificity LocAcc | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Figurines | 0.384463 | 0.393214 | +0.008751 | 0.678571 | 0.696429 | +0.017857 |
| Ramen | 0.283838 | 0.287548 | +0.003710 | 0.535211 | 0.521127 | -0.014085 |
| Teatime | 0.374127 | 0.375740 | +0.001613 | 0.542373 | 0.559322 | +0.016949 |
| Waldo Kitchen | 0.216636 | 0.222037 | +0.005400 | 0.545455 | 0.590909 | +0.045455 |
| Scene macro | 0.314766 | 0.319635 | +0.004869 | 0.575403 | 0.591947 | +0.016544 |

The candidate improves mIoU on 4/4 scenes and localization on 3/4. Ramen has
a 0.0141 localization decrease, but its mIoU improves and no severe scene
regression occurs. The fixed specificity rule therefore replaces max in the
v4 candidate; max remains the frozen ablation and the last v3 behavior.

## Artifacts

- ScanNet candidate results:
  `output/optimization_20260724/text_specificity_margin002/scene*_00.json`
- LERF unaries:
  `output/optimization_20260724/text_specificity_margin002/*_unary_specificity002.pt`
- GPU queue:
  `radio_gs/scripts/run_lerf_specificity002_eval_queue.sh`
