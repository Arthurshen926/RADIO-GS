# Positive coverage under compact selected-set inference (next97)

2026-10-02. All8scene evaluations complete. Balanced pixel loss plus the existing
high-resolution evidence/footprint prior reaches61.820% selected projection IoU,
versus next96 61.478 and next93 60.613. Preserve as an optional higher-mean LERF
research candidate. Not a global default or a complete-object guarantee.

## Mechanism and direct evidence

The previous source BCE averaged all known pixels together. Across the13refit
queries, positive source pixels account for only0.16–1.43% of known pixels.
The overlap term is already present, but it does not make BCE class-balanced.
The previous footprint prior can therefore discard some supplied positive support.
This is an objective-design limitation under the measured imbalance, not a claim
that every mean-BCE implementation is a software bug.

Keep all masks, confidence, unknown regions, native scores, geometry, initialization,
source views, sampling and optimization steps fixed. Replace the mean BCE with:

L_pixel = 0.5 * mean_positive(BCE) + 0.5 * mean_negative(BCE).

Keep the same overlap loss and footprint-prior coefficient0.1. No penalty sweep,
new teacher, source identity change or per-query choice. Compare the same balanced
loss with and without footprint regularization on the high raster. This is a soft
positive emphasis, not a hard coverage constraint or a completeness guarantee.

After fitting, render each hard selected set in every active source view with the
same high raster/interleaved pixels and alpha threshold10.5/255. Compare the
candidate to next96 on the exact same positive/negative masks. This diagnostic is
not used to select checkpoints or train and uses no target GT. Views are reused
training sources, not independent validation. Coverage is against teacher evidence,
not verified physical foreground.

## Source-positive coverage

Fraction of known positive pixels covered by the hard rendered subset:

| Query | Positive fraction among known | next96 | Balanced+footprint |
|---|---:|---:|---:|
| waldo | 0.199% | 85.34% | 94.08% |
| tesla door handle | 0.646% | 86.08% | 97.02% |
| pikachu | 1.426% | 95.08% | 98.85% |
| sake cup | 0.855% | 85.74% | 92.60% |
| spatula | 0.203% | 96.78% | 97.21% |
| plastic ladle | 0.160% | 97.97% | 98.98% |

Thus some of the lost positive evidence is recoverable by changing the objective
without new geometry or source information. The remaining source coverage gap can
also involve representation/optimization/inconsistent evidence; it is not uniquely
attributed here. A higher source coverage value cannot certify target identity.

## Frozen evaluation

Selected Gaussian silhouette projection IoU (%):

| Scene | next96 | Balanced+footprint | Balanced without prior |
|---|---:|---:|---:|
| figurines | 65.982 | 67.159 | 67.267 |
| ramen | 55.303 | 54.740 | 52.835 |
| teatime | 63.673 | 63.809 | 62.166 |
| waldo_kitchen | 60.953 | 61.572 | 58.127 |
| Macro | 61.478 | **61.820** | 60.099 |

Gain is+0.342points versus next96 and+1.207 versus next93. Three scenes improve
versus next96, ramen regresses. All four improve versus next93, but many individual
queries still regress against that older candidate. The without-prior matched
control is next95 high60.256→60.099, not an improvement. Both preserving positive
support and limiting extraneous extent matter in these experiments.

| Query | next96 | Balanced+footprint |
|---|---:|---:|
| pikachu | 81.63 | 91.69 |
| tesla door handle | 65.51 | 71.49 |
| waldo | 56.59 | 67.88 |
| kamaboko | 78.38 | 81.33 |
| onion segments | 0.00 | 0.00 |
| sake cup | 67.03 | 56.94 |
| dall-e brand | 49.59 | 50.43 |
| paper napkin | 79.86 | 81.13 |
| ketchup | 69.54 | 73.25 |
| ottolenghi | 63.45 | 69.28 |
| plastic ladle | 0.00 | 0.00 |
| pour-over vessel | 86.68 | 89.51 |
| spatula | 38.90 | 40.17 |

Waldo56.59→67.88 is a partial recovery, still below next93 78.22. Its target recall
57.2→69.4%, precision98.3→97.0%. Spatula38.90→40.17 has precision42.3→43.0% and
recall82.9→85.8%. Pour-over86.68→89.51 has precision99.5→97.3%, recall87.1→91.7%.

Sake cup is a material failure:67.03→56.94, while source-positive coverage rises
85.7→92.6%. Target precision81.7→67.1%, recall68.1→69.4%. More teacher-positive
coverage is NOT equivalent to correct object extent; the source labels/correspondence
or their transfer may support excess scope. This experiment does not uniquely
identify which one. Onion/ladle remain0 despite improved source-mask coverage.
Do not claim their identities have been repaired.

All8outputs preserve native semantic scores and the other54 query selections
exactly. Semantic IoU66.930/AP84.146 unchanged. This is2D silhouette projection,
not a new independent3DGT metric. Other benchmark tracks were not rerun.

## Decision and next focused work

Keep high balanced+footprint as an optional research candidate; retain next96 and
next93 for comparison, next80 stable reference, defaults unchanged. No per-query
winner stitching. No claim of large benchmark-wide or publication-ready gain.

The experiment separates two causes of low recall/precision:
1. Existing source positives can be underweighted. Balanced BCE partly repairs
   this and recovers actual held-image target recall on Waldo/handle/pikachu.
2. Other errors persist or grow even when source positives are well reconstructed.
   They require checking the meaning and cross-view validity of the source scope,
   not simply higher positive weight, weaker regularization or more steps.

Next isolate candidate identity versus scope transfer for source-fit/high-error
queries using source RGB and views excluded from correspondence/hypothesis decisions.
For sake cup, inspect the positive regions whose stronger fitting increases target
false extent; do not use target labels to select a source hypothesis. For Waldo,
separate missing positive scope from supplied-but-unreconstructed support. These
are pending work, not conclusions that supervision errors have already been fixed.

The code remains the optional source-assisted query-time extent path. This does
not validate a generalizable compact field or unified text/image/2D/3D prompts.
No new NVOS/ScanNet/UQIS, independent-test, independent3DGT or SOTA claim.

## Reproduction and artifacts

run_coverage_scope_next97.py --scene SCENE --gpu GPU
  --arms high_footprint high_none

summarize_coverage_next97.py; measure_coverage_errors_next97.py;
measure_coverage_sets_next97.py.

- output/next97/summary.json (matched mean-loss baseline per arm).
- output/next97/source_coverage.json; error_decomposition.json; selected_sets.json.
- output/coverage_scope_next97/<arm>/<key>/fit.json records source coverage and
  unchanged fitting conditions; typed_readout.pt and evaluation.json record results.
- configs/recipes/next97_research.json.

Decoder pixel-loss defaults to mean. Optional source-coverage-reference enables
post-fit diagnostics only; it does not supply supervision or initialize fitting.
