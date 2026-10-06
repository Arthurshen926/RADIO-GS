# Category-set discovery: transfer and same-evidence solver controls (next107)

## Question and scope

Next106 recovered a tiny, precise onion instance and improved its category-set
recall using text plus an automatic visual example. This experiment asks whether
that mechanism transfers beyond this one development query, whether separate
component examples help, and whether the downstream set solver contributes beyond
teacher evidence itself. These are full four-scene LERF development controls,
not new NVOS/ScanNet/UQIS results or an independent-test/SOTA claim.

All13previously diagnosed difficult queries are updated, with the remaining54
selection columns inherited unchanged. Use **the same** next104 core/merged tracks
for every arm. The matched parent is `core_scope_next104/core`,62.236484%macro,
not the selectively retained next104 candidate62.598590%. In particular, this
transfer experiment keeps the old erroneous onion seed rather than silently
inserting next106's improved seed; it isolates discovery from anchor replacement.
No per-query ground-truth winner selection is used.

## Implemented method controls

Shared frozen geometry, source cohorts293/122/169/180, native field/query space,
calibration, renderer and selected-mask evaluator are unchanged. Within each
originally core-present source view, use full source RGB and exact query text.
Image features are reused across queries/arms within a frame.

- **Text:** preserve the core and union SAM3 text-only discoveries.
- **Box:** same text plus one normalized box enclosing the original positive core.
- **Components:** same text plus every8-connected original core component with
  at least one3x3eroded positive pixel, each as a separate positive box. There is
  no arbitrary cap on component count. If none qualify, this arm is text-only.
- **Self-supported:** reuse Box detections; weight additional probability by mean
  detector-union probability on the original core, then take its maximum with
  the core. This tests prompt compliance, not semantic truth. There is no new
  teacher call or fitted threshold.
- **Self-supported/unknown:** same weighting, but original detector-positive
  extensions are excluded from negative supervision if their probability is
  subsequently reduced. Positive permission remains unchanged and view-level
  reliability remains1. This distinguishes withholding belief from declaring
  background. The probability-only arm inherits the decoder's ordinary complement
  handling, so it cannot claim that suppressed extensions remain unknown.

Discovery never recursively promotes its new detections into examples. Region
instances are stored separately in transient caches, then combined for the current
category-set task. This is not a physical-instance association model, and this
union must not be reused as an NVOS/reference-instance query contract.

All reconstruction arms use source initialization, balanced positive/negative
pixel loss, corroborated negatives,192x256source raster/interleaved rays and
footprint prior0.1. They rebuild their own supervision and optimization measure;
more masks can change initialization, active views and support-cost normalization.

A separate **same-evidence source-fusion control** uses Box's exact evidence and
admission/camera operators, returning the confidence-weighted positive fraction
from source backprojection with prior values on unknown support. The new explicit
`--source-fusion-only` option skips subset-render optimization. The optimized
comparison includes the renderer, optimizer and footprint regularization together;
it is not an isolated proof that one loss term, or the renderer alone, explains
the gain. This baseline uses the same native semantic field and fixed selection
threshold as the optimized path.

## Completed results

All24new scene evaluations completed; entries are percentages.

| Arm | Macro | Figurines | Ramen | Teatime | Kitchen |
|---|---:|---:|---:|---:|---:|
| Matched core baseline | 62.2365 | 67.1027 | 56.4379 | 63.8332 | 61.5722 |
| Retained next104 | 62.5986 | 67.4508 | 56.9085 | 63.7335 | 62.3015 |
| Text-only discovery | 62.6333 | 67.1051 | 56.4379 | 63.8332 | 63.1572 |
| Text + one core box | 63.1135 | 67.1299 | 57.5343 | 63.8482 | 63.9415 |
| Text + component boxes | 63.0701 | 67.2103 | 57.5650 | 63.8370 | 63.6681 |
| Same Box evidence, source fusion only | 57.5694 | 64.6684 | 49.6639 | 60.1672 | 55.7782 |
| Prompt-compliance weighting | 62.6114 | 67.1379 | 57.3970 | 63.8253 | 62.0853 |
| Compliance + unknown extensions | 62.6148 | 67.1546 | 57.3970 | 63.8253 | 62.0822 |

Box exceeds matched core by0.8770percentage points, retained next104 by0.5149points, and its same-evidence source-fusion control by5.5441points. Self-supported/unknown adds only0.0034points over probability-only compliance and neither beats Box. Native semantic tensors and all54untouched selection columns are exact in every new output.

The Box arm improves every scene against the matched core baseline. Compared
with retained next104, three scenes improve but figurines regresses. Separate
component boxes do not beat the single-box macro. The scope branch preserves
native semantic scores exactly; this is **not** a new2Dsemantic-IoU improvement.

Selected per-query IoU(%, matched core → Box; retained next104 in parentheses):

- plastic ladle:0→48.87 (0). Text-only already reaches47.02: most of this recovery
  comes from newly consumed independent text detections, not visual examples.
- sake cup:77.03→87.09 (77.03).
- ottolenghi:69.28→88.25 (71.21).
- ketchup:73.25→79.40 (78.60).
- spatula:40.17→16.49 (49.36), a severe regression.
- tesla door handle:69.92→67.67 (69.92).
- onion segments:0→0; changing discovery does not repair its wrong egg anchor.

The full per-query result matrix, all scene means, source discovery counts and
unchanged-output comparisons are in `output/next107/summary.json`.

## Why a mean gain is insufficient

Source illustrations select the frame with the most newly proposed pixels for
each query, using no target annotation. They are deliberate failure-oriented
examples, not a representative accuracy estimate. In ramen, a weak peripheral
sake core can cause a bowl-sized discovery. An egg anchor for onion generates
more eggs. A consistency check that merely reproduces its own prompt cannot
establish correct identity; conversely, an incorrect original core can reject
useful independent text discoveries. Self-supported weighting illustrates this:
it reduces the plastic-ladle gain substantially.

Multiple connected components are also not automatically complementary semantic
examples: they can be fragmentation of one object, wrong background, or unrelated
instances. This explains why adding component boxes is not yet a validated
multi-example identity method. The alternative requires retaining hypothesis
identity and evidence type until set inference, rather than treating every
accepted pixel as interchangeable positive evidence.

## Decision, claim and next mechanism

Keep Box as a uniform **optional higher-mean development candidate**, with its
regressions visible; do not replace the stable default or construct query-wise
best-of outputs. Next104 remains the rollback candidate, next80 the stable recipe.
No claim is made that the proposed visual prompting itself is new foundation-model
capability. The same-evidence projection comparison supports the practical value
of explicit set optimization; it does not establish novelty, compact student
learning, full multimodal generalization, or submission readiness.

The next focused mechanism must separate **anchor identity**, **independent text
identity evidence**, and **set extent**. Allow independent detections to correct
an erroneous anchor without permitting unrelated expansion. Preserve distinct
instance/part hypotheses through cross-view association and inference instead of
flattening them into one union early. Validate the resulting unified rule across
all13queries, including spatula, before widening to allqueries/benchmarks. Do not
route individual queries to this round's best controls.

For a CVPR claim, the remaining evidence is substantial: an independent validation
split; same-input/cost strong baselines; semantic-field and compact-student
contribution under fixed evidence; typed image/2Dprompt/3Dprompt transfer; and
full current-map NVOS/ScanNet evaluation after a general rule is established.
UQIS remains paused under the user's access constraint. Prior benchmark results
must not be relabelled as evaluations of this new source-discovery method.

## Reproduction

`run_set_discovery_next107.py --scene SCENE --gpu GPU` builds the three matched
source arms and evaluates each. `run_source_fusion_next107.py` evaluates the
same-evidence backprojection. `run_self_supported_next107.py` and
`run_self_supported_unknown_next107.py` evaluate the two compliance controls.
`view_set_discovery_next107.py` creates source-only illustrations;
`summarize_sets_next107.py` reports every arm, query and scene.

The initial unknown-expansion launch omitted required scope-reliability metadata;
that run produced no result. The builder was corrected to carry explicit1.0
reliability, preserving the matched view weights, and all affected jobs rerun.
No unit tests or external issue publication were added. Source discovery requires
extra online SAM3 calls and source RGB; cached multi-arm wall time is not a fair
per-query latency measurement and no low-latency claim is made.
