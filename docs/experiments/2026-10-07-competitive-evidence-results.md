# Relative signed evidence: small fern gains, no common promotion

The [fixed protocol](2026-10-07-competitive-evidence-protocol.md) is complete.
Six frozen exports, three complete signed teacher cohorts, six raw fern prompt
predictions and scores, six four-input export replays and the full four-scene
unsigned LERF audit are finished. No fitting or parameter search was performed.
The [machine-readable evidence](2026-10-07-competitive-evidence-results.json)
binds configuration, code, exports, predictions, replay and failed gates.

**Keep the single-support fitted physical field and product-negative default.**
Ray-mixture competition improves the three fern public readouts slightly but
misses the fixed figurines teacher-subset gate. Pooled competition also lowers
fern AP and subset IoU. Neither triggers the second NVOS task. These small
fixed-field differences are not claims of statistical significance; the strict
predeclared gate is preserved rather than relaxed after viewing outcomes.

## Common operator and invariants

Positive, negative and null evidence compete with scores `exp(l+), exp(l-), 1`:
`sigmoid(l+ - softplus(l-))`. The implementation retains unsaturated positive
logits. Both signs use the same member kernel, physical scale and protected
native coordinates. Scores are not calibrated posteriors.

The pooled control retains the previous negative prototype. The ray control
uses each ray's accepted normalized group prototype and physical centre, then
mixes exponentiated logits by observed evidence mass. Every ray participates in
fixed chunks of 32. It uses neither the failed expected-energy variance penalty
nor independent maximum vetoes. Positive raster pooling remains exact.

The mixture's algebra is invariant to splitting identical evidence or repeating
the entire measure. This is not invariance to an actual raster change, which can
change geometry associations. Nor does averaging guarantee monotone exclusion
when a different negative observation is added. It is a tested research score,
not a complete model of interactive constraints or association uncertainty.

All learned tensors, geometry, native coordinates, scales and thresholds remain
unchanged. No source tracking, query-time mask generator, extra teacher, 3D
labels or dataset-specific decision enters the operator. Old manifests default
to `negative_rule: product`; new rules remain explicit opt-in controls. Without
negative evidence, the existing candidate computation and scores remain exact.

## Complete fixed outcomes

Fern uses one unchanged physical export, registered raw official scribbles,
default scope and the retained scorer. All six predictions and prompt-agreement
receipts precede target scoring; all six pass copied-field source-denied replay.

| Signed rule | Response IoU (%) | Response AP (%) | Actual subset IoU (%) | Selected Gaussians |
|---|---:|---:|---:|---:|
| Parent product | 76.7399 | 92.0952 | 76.1453 | 121,865 |
| Pooled competition | 76.7403 | 92.0907 | 76.1329 | 121,892 |
| Ray-mixture competition | 76.7569 | 92.1666 | 76.4433 | 121,284 |

Ray competition changes response IoU/AP/actual IoU by +0.0171/+0.0714/+0.2980
percentage points relative to the signed parent. The saved v10 fern response
is 82.1851%, with different source-access, prompt and construction budgets; the
unpromoted experiment leaves a 5.4282-point response gap. This is 1/8 previously
opened NVOS cases, not an aggregate NVOS gain or an independent final test.

All three positive-only arms are bit-exact at response IoU/AP/actual subset
76.7395/92.0912/76.1066%, selecting 121,929 Gaussians. Their signed variants keep
the same 99.9396% reference positive coverage and 0.5073% negative selected
coverage. Both new rules pass this prompt screen. The ray rule removes 645
Gaussians from the positive-only set, compared with 64 under the parent.
Agreement at sparse authorized strokes does not establish complete object scope.

Every source-development SAM region uses its original centroid-interior ray and
up to four uniformly spaced known negative pixels. These are default-scale
teacher-mask diagnostics, not independent benchmark labels. All cases remain in
the denominator; fern has two regions without usable negative pixels. There are
no unsupported signed 2D cases in this cohort.

| Scene / cases | Product subset IoU (%) | Pooled competition | Ray competition |
|---|---:|---:|---:|
| figurines / 169 | 37.4796 | 37.4474 | 37.4554 |
| ramen / 149 | 40.3822 | 40.3425 | 40.4172 |
| fern / 33 | 44.5648 | 44.6790 | 44.9112 |

Ray competition misses the figurines gate by 0.0242 points. Pooled competition
misses figurines and ramen. Their normalized member-response teacher means rise
in all three scenes; this again does not guarantee improved actual selected sets.
The public fern gains cannot override the declared common-method gate.

## Source-only identity and extent audit after the study

A separate post-study audit examines every original teacher-region image query,
without public target masks or new optimization. It first reproduces all saved
image/2D/3D parent IoUs exactly. For the image-inferred anchor, it measures the
fraction of that Gaussian's visible opacity contribution inside the source mask.
At least half counts as target-supported; no visible contribution is unknown,
not a wrong identity. This is per-view mask association, not whole-object 3D truth.

| Scene | Target-supported anchor | Visible off-target anchor | Not visible | Image IoU when target-supported (%) |
|---|---:|---:|---:|---:|
| figurines | 77 / 169 | 39 / 169 | 53 / 169 | 42.6501 |
| ramen | 102 / 149 | 21 / 149 | 26 / 149 | 39.8360 |
| fern | 31 / 33 | 0 / 33 | 2 / 33 | 44.2962 |

For the visible off-target group, image -> registered 2D IoU is
7.4383 -> 30.9693% on figurines and 15.0558 -> 28.2333% on ramen. Supplying a
region-interior ray helps these cases, while correctly associated image anchors
still leave considerable member-extent error. This does not identify a unique
cause or provide an oracle upper bound.

Only 11.74/16.15/1.83% of image false-added pixels overlap current comparable-scope
known exclusions on figurines/ramen/fern. These are pooled pixel counts over all
teacher cases; repeated pixels in different queries are separate observations.
The rest lie outside the complete diagnostic SAM target but remain unknown to
the current partial-supervision recipe. They are not automatically background.
Even within the target-supported-anchor group, known-exclusion overlap is only
4.62/15.27/1.93%. Stronger negative aggregation alone cannot create the missing
verified exclusions. This refines, rather than replaces, the earlier train-only
false-addition audit.

The next architectural experiment should retain identity evidence while learning
object extent through the same member operator, and test source-view consistency
of that extent. Candidate identity and member scope must be measured separately.
The current native vector is used to select anchors, after which membership
depends on their grouping features and geometry; preserving the identity
condition beyond that selection is a hypothesis to test, not a validated fix.
Do not repeat support-count/negative-strength/threshold sweeps or turn unknown
source pixels into negatives. No per-input decoder or inference source bank is
justified by these findings.

## Verification and reproduction

63 distinct tests pass: 59 field/basis/encoder tests plus four supplemental signed
text/image instance/union checks. All six exports have exact parent state tensors
and unchanged tensor capacity. Six real-encoder text/image/2D/3D copied-field
replays pass. All six raw NVOS payloads replay exactly on their original GPU 5,
including selected rows, memberships, selected alpha and normalized response.
The Python open-denial audit is not an OS sandbox. Previously observed cross-GPU
renderer and non-bitwise training limits remain; this study contains no fits.

All 68 LERF query vectors (21/15/14/18 across the four scenes), for all three
policies, reproduce saved native responses, memberships and selected sets
exactly. No LERF target masks need reopening. The accepted four-scene actual
subset mean stays 32.3130%, versus saved v10 66.9633% under different contracts.
Neither full five-benchmark success nor compactness has been established.

The ray rule's one shared-GPU fern decode took 0.704s versus parent signed 0.463s;
peak allocated memory was 2.075GB versus 1.897GB. These single-run timings are not
a controlled latency comparison. No deployment tensors were added.

Preparation/execution: `scripts/research/{prepare,run}_competitive_evidence.py`,
`run_competitive_evidence_nvos.py`, `audit_query_field_identity_extent.py`, and
`summarize_competitive_evidence.py`; configurations live under
`configs/query_field/competitive_evidence*`. Use the existing cybersim_agent
environment and `PYTHONPATH=.`. Outputs under
`output/query_field_20261007/competitive_evidence_v1` refuse overwrites. Choose
a fresh run root for reproduction; never regenerate sealed receipts in place.
Original-worktree branch, HEAD and index remain unchanged. All launched jobs
are complete; no additional NVOS scene, refit or hidden parameter search follows.
