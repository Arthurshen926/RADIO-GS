# Independent source extent calibration — next55

## Question and decision

Can the next54 native-field ranking recover useful extent without borrowing the
next53 confidence distribution? Two full-four-scene source-only controls completed.
Neither replaces next54. This round does **not** deliver the requested large gain.
No default was changed and no new benchmark beyond LERF development was evaluated.

## Mechanism

Keep next52 geometry, next54 mixture_scope responses, query encoders, source
frames/detections and selected membership fixed. Fit a single nondecreasing
probability map shared across queries within each scene, using weighted
pool-adjacent-violators with 256 input bins. Source eroded positives and dilated
mask-complement pseudo-negatives provide labels; missing detections and boundary
bands abstain. Query-view contributions are normalized before detector confidence.
No target images or target labels enter fitting. These are existing development
queries and online source evidence, not an unseen-query offline reconstruction.

* Pixel: fit the mapping to source-rendered responses, then apply to elements.
* Element: backproject weighted positive and total evidence through the contribution
  operator before fitting against element responses. This addresses the calibration
  domain mismatch explicitly, but is not exact nonlinear inverse rendering.

Interpolation preserves nondecreasing element order; flat portions can create ties.
Rendering mixtures means pixel ranking is not invariant. Neither variant removes
other dependencies on old artifacts or changes actual set membership.

## Frozen evaluation

Four-scene macro, percent; same scorer and declared output contracts:

| Method | 2D IoU | AP | Localization | Selected projection IoU |
|---|---:|---:|---:|---:|
| Retained next54 | 58.087 | 74.965 | 82.245 | 44.037 |
| Source pixel monotone calibration | 51.891 | 75.805 | 82.691 | 44.037 |
| Source element monotone calibration | 51.081 | 75.740 | 83.827 | 44.037 |

Both lose IoU on every scene versus next54. Higher AP/localization is a tradeoff,
not overall replacement evidence. Selected IoU is unchanged by construction;
it is selected Gaussians projected against 2D annotations, not physical 3D GT.

| Scene | Retained IoU | Pixel IoU | Element IoU |
|---|---:|---:|---:|
| figurines | 58.088 | 52.208 | 48.063 |
| ramen | 48.903 | 43.177 | 42.229 |
| teatime | 68.197 | 61.285 | 61.352 |
| waldo_kitchen | 57.161 | 50.892 | 52.681 |

## What is actually localized

Source-known confidence-weighted Brier error improves relative to retained next54
on figurines (0.002503→0.002437), ramen (0.003816→0.003704), and teatime
(0.002987→0.002740) for pixel calibration, while real IoU falls. Waldo source
error worsens (0.002593→0.003331), so this is not a universal source-fit gain.
Element calibration shows the same three-versus-one pattern.

Source-positive recall is also lower than retained next54: pixel calibration
71.86% vs78.46% on figurines and86.64% vs90.20% on ramen. Its source-negative
specificity is higher. Thus even known positives can be traded away while the
background-dominated aggregate error looks good. These are pseudo-label metrics,
not true recall or evidence that source target completeness is known.

This rules out these two monotone calibration replacements as sufficient. It does
NOT prove that all remaining loss is caused by incomplete masks: loss weighting,
shared-query calibration, noncommuting rendering/calibration and source-to-target
geometry can contribute. Earlier balanced/per-query controls also failed; simply
repeating those parameterizations is low priority.

## Consequence for the research plan

The key unmet capability remains independently estimating a target's full extent,
not merely fitting existing responses more closely. Do not use source Brier as a
model-selection proxy or advertise another calibration layer as a contribution.
The next substantive intervention should supply separately qualified positive
identity/extent and negative-exclusion evidence, and validate actual subset
rendering alongside semantic rendering. A held-out source-view comparison is
needed to distinguish fit from transfer; detector complement alone cannot certify
complete exclusion. Existing reviewed scope materials may inform this work but
assistant-reviewed labels must not be called human GT.

Keep the native semantic branch and separate selected-set decoder. A new method
must improve them under one fixed recipe, rather than selecting scene/query
winners. Compact four-teacher reconstruction, independent generalization, fair
strong-baseline comparison and deployment cost remain open CVPR requirements.

## Artifacts

* `scripts/research/calibrate_source_extent_next55.py`
* `scripts/research/run_source_extent_next55.py --domain pixel|element`
* `output/source_extent_next55/summary.json`
* Per-scene `fit.json` records source diagnostics and fitted monotone blocks.

All eight fitting/evaluation combinations completed. Diagnostics were then
recomputed with the identical fitting recipe; only reporting was added, not a
new intervention. No defensive tests or audit framework was introduced.
