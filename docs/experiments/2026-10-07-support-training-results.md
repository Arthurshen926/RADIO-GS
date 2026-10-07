# Matched support-form training: complete negative result

The [fixed protocol](2026-10-07-support-training-protocol.md) compares two
source-only training changes on figurines, ramen and fern. **Neither passes the
common-method gate. Keep the fitted physical member field as the candidate.**
This round does not reduce its aggregate gap to v10. It adds a shared signed
training path, a controlled failure and a more specific representation issue;
it does not add a benchmark-specific decoder or inference source access.

All six fits, their teacher diagnostics, four LERF prediction/scoring jobs and
four real NVOS prompt jobs are complete. All eight public predictions precede
new target scoring. Source-denied real NVOS replays and four-input LERF smoke
checks pass. Forty distinct tests pass. The
[evidence ledger](2026-10-07-support-training-evidence.json) binds code and artifacts.

## Intervention and controls

The original recipe fits a single positive dominant Gaussian per source episode.
The unsigned arm cycles a single anchor, one uncertain ray, four positive rays,
and four positive rays. The signed arm changes the last slot to four positive
and four negative rays. Every support uses the deployed deterministic ray adapter
and the same observed-support filter/member product. Negative pixels come only
from the original disjoint comparable-scale source masks; unknowns stay unknown.

All arms retain the same geometry/native tensors, feature views, 2000 optimizer
steps, random initialization, physical scales, candidate policy and thresholds.
A separate support RNG leaves the original episode/first-pixel draws untouched.
The complete 2000-step traces match between unsigned and signed arms; logged
episode ordering matches the parent. Every signed fit has 500 valid signed
episodes and zero missing-negative fallbacks. Unsigned/signed fits make
2000/2500 member evaluations respectively, a 25% member-compute difference during
training; each still has 2000 rendered losses and optimizer updates.

There are no new inference parameters or field tensors. Export bytes remain
2,658,118,304 / 2,841,450,464 / 1,577,411,680 for figurines / ramen / fern.
These are high-capacity references, not compressed students. Wall times vary
with loading and shared GPUs and are not a controlled efficiency comparison.
The default training policy remains `single`; both new policies are opt-in.

## Actual benchmark outputs

| Scene / readout | Common physical parent % | Mixed unsigned % | Mixed signed % |
|---|---:|---:|---:|
| Figurines actual LERF subset IoU, 56 query-views | 41.7884 | 33.5579 | 34.3990 |
| Ramen actual LERF subset IoU, 71 query-views | 19.4642 | 22.1834 | 20.7247 |
| Fern positive response IoU | 76.7395 | 76.2668 | 76.2888 |
| Fern signed response IoU | 76.7399 | 76.2658 | 76.2887 |
| Fern signed response AP | 92.0952 | 91.0246 | 91.3218 |
| Fern signed actual subset IoU | 76.1453 | 75.2162 | 75.2016 |

Ramen improves in isolation; figurines loses substantially and fern regresses in
all three required real signed readouts. Selecting a different arm per scene
would conceal the failure. No teatime/waldo confirmation is triggered because
neither arm passes the initial gate. The accepted four-scene LERF actual subset
mean stays **32.3130%**, and its saved v10 comparator stays **66.9633%** under
different source/resource conditions. The fern accepted response gap remains
5.4453 points to saved v10 under different raw-prompt/source-access conditions.
No full NVOS, ScanNet OVS or UQIS progress is claimed from these tests.

Native text semantic responses are bit-exact. Member semantic IoU drops to
43.9758/40.1774% on figurines and 38.6341/37.5323% on ramen for unsigned/signed,
versus parent 49.1988/42.8233%. Native preservation alone therefore does not
guarantee query capability after the learned member readout.

Retained-loader replay verifies every saved LERF subset IoU before decomposing
it. Figurines unsigned/signed precision is only 33.8121/34.7529%, while recall
is 97.9759/96.8112%; the low precision remains the major observed symptom.
Ramen unsigned/signed precision is 23.4708/22.0552%, recall 83.6091/83.8688%.
These statistics describe the outputs; they do not uniquely identify a cause.

## All default teacher-input diagnostics

These are SAM-region development diagnostics, not real external-image or human
3D-point results. Every region, including unsupported cases as zero, is retained.

| Scene / arm | Image subset IoU % | 2D subset IoU % | 3D subset IoU % |
|---|---:|---:|---:|
| Figurines parent | 23.6868 | 37.4463 | 32.2285 |
| Figurines unsigned | 23.5537 | 38.9448 | 32.6958 |
| Figurines signed | 24.1706 | 39.1942 | 33.1340 |
| Ramen parent | 31.7428 | 40.3042 | 38.8454 |
| Ramen unsigned | 32.2206 | 40.8166 | 39.5857 |
| Ramen signed | 33.4344 | 41.0142 | 38.6463 |
| Fern parent | 43.4368 | 44.5748 | 44.1197 |
| Fern unsigned | 43.8035 | 42.7357 | 42.0853 |
| Fern signed | 44.4215 | 42.4084 | 42.3621 |

Figurines has 169 cases per input, ramen 149 and fern 33. The inherited figurines
unsupported 3D case remains in the denominator. Better diagnostics on some input
types cannot override the public subset and remaining input failures.

## Training repeatability versus query replay

The single-policy refactor replay exactly matches the original initialization
and logged episode order, but not final learned tensors. Two further runs of
the exact b831d03 field/fitter also produce different final tensors with identical
initialization. Thus bitwise training reproducibility was not established even
before this refactor; the observation is not specific evidence against the new
support implementation. The exact CUDA cause has not been isolated.

The refactored singleton replay and two exact-old-code replays have fern
image/2D/3D means near 43.30/44.42/43.93%. Their differences are under 0.005
percentage points here, while the historical saved fit is up to about 0.19 points
different. Small deltas should not be called statistically established gains.
This three-run, one-scene check is not a multi-seed confidence interval, nor a
bound transferable to other scenes. It does not explain away the much larger
figurines text and fern point regressions.

By contrast, fixed deployed models replay exactly: four copied NVOS model/prompt
combinations reproduce member scores, selected rows and both renderings, and
each of the four LERF exports passes real text/image/2D/3D adapter replay. Original
source/output opens are denied by Python file-open hooks, not an OS sandbox.

## What this narrows down

Merely replacing most singleton training episodes with pooled/signed episodes is
insufficient. It also reduces direct singleton training from 2000 to 500 episodes,
so the study does not isolate support diversity from that redistribution of
training pressure. No mixture ratio, support count or threshold sweep followed.

A separate **post-study, source-only** audit examines 32 uniformly spaced eligible
source episodes per scene using the unchanged parent field. The mean norm of the
weighted group vector before prototype normalization is:

| Source scene | Single uncertain ray | Four positive rays | Four negative rays |
|---|---:|---:|---:|
| Figurines | 0.7684 | 0.7280 | 0.4798 |
| Ramen | 0.8348 | 0.7279 | 0.4739 |
| Fern | 0.9159 | 0.8709 | 0.7788 |

The current member kernel normalizes every such vector to unit length. It
therefore discards directional disagreement, and a flat negative pool also loses
which background constraints came from distinct rays. These norms are geometric
statistics, not calibrated probabilities or proof that restoring them improves
accuracy. They identify a concrete representational limitation worth testing.

The next method refinement should distinguish uncertain alternatives *within* a
registered ray from multiple positive observations and independent negative
constraints, while preserving one common conditional member kernel. Validate
aggregation algebra on controlled cases, then fit it using source-only evidence
and retain singleton competence. Evaluate actual hard subsets and normalized
responses together; repeat consequential fits before claiming small gains. This
is a more specific architectural question than changing support-mixture weights,
and introduces no authorization for source-query fallback or extra 3D labels.
