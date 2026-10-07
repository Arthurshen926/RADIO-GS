# Expected support energy and independent negative rays: rejected controls

The [fixed study](2026-10-07-expected-support-protocol.md) is complete: six
source-only fits, three frozen controls, nine complete teacher-development
evaluations, six LERF predictions/scores, six real NVOS predictions/scores and
nine four-input export replays. **Neither fitted arm passes. Retain the accepted
physical field; this study does not close its aggregate gap to v10.**

The [evidence ledger](2026-10-07-expected-support-evidence.json) binds code,
exports, source traces, scores and replay limits. No per-scene/input winner is
selected. No teatime/waldo extension or additional fit is triggered. All launched
jobs are complete; ScanNet OVS/UQIS and full NVOS are not evaluated here.

## Intervention and controls

The source compiler, geometry, protected native coordinates, source labels,
inferred-anchor domain, physical scale, thresholds and singleton kernel stay
fixed. Expected energy keeps the weighted group resultant length and adds the
exact within-support spatial variance to the physical energy. It equals the
weighted average of singleton logits, including the optional paired native
metric. It is **not** an average of sigmoid responses or a calibrated
latent-association posterior.

Negative supports retain ray identities. Each ray averages its uncertain
contributors in energy; the largest ray exclusion multiplies the positive score.
This is monotone under extra negatives and idempotent under duplicate whole
constraints. All rays participate in chunks of 32. Fractional raster evidence
can still supply a full-strength maximum constraint.

Both fits use 2000 paired updates from the original initialization: the original
singleton task and an auxiliary ray task receive half loss weight each.
Auxiliary forms cycle one ray, four positives and four positives plus four
negatives. They retain 2000 singleton presentations but double rendered losses
to 4000; singleton gradient weight is not preserved. Their complete episode,
first-pixel and support traces match. All six have 667/667/666 auxiliary episodes
and zero missing-negative fallbacks. Legacy uses 4666 member conditions;
expected energy uses 6660/6662/6664 on figurines/ramen/fern because some source
exclusions have fewer than four distinct pixels.

No new learned parameters or scene tensors are introduced. Exports remain about
2.658/2.841/1.577 GB, with shared input encoders additional. Training times,
including loading on shared GPUs, are 198.98/248.53/224.56 seconds for legacy
and 117.80/180.59/154.20 for expected energy. Their order is not a controlled
speed comparison. The new policies remain opt-in; defaults remain singleton
training and normalized-prototype aggregation.

## Complete public outcomes

Actual LERF subset IoU uses the unchanged mask loader and alpha-byte threshold
above 10. There are no missing predictions: 56/71 query-views respectively.

| Scene / output (%) | Accepted parent | Frozen expected | Paired legacy | Paired expected |
|---|---:|---:|---:|---:|
| Figurines actual subset IoU | 41.7884 | 41.7884 | 38.4655 | 25.3824 |
| Ramen actual subset IoU | 19.4642 | 19.4642 | 24.5118 | 19.1874 |
| Figurines member response IoU | 49.1988 | 49.1988 | 51.2627 | 56.4633 |
| Ramen member response IoU | 42.8233 | 42.8233 | 48.8693 | 47.1673 |

Frozen text outputs are bit-exact because inferred supports are single elements.
All native semantic responses are also bit-exact. Higher member response IoU
does not imply cleaner selected geometry. Figurines paired-expected actual
subset precision/recall is 25.5885/99.0131%, versus parent 42.1660/98.2269%.
Paired legacy gives 38.9694/98.1012%. Ramen precision/recall is 20.3050/84.1841%
for expected and 25.7205/84.2050% for legacy, versus parent 20.7656/84.0656%.
Every decomposition first reproduces the retained per-mask IoU exactly.

Real fern uses unchanged raw registered prompts, default scale, normalized
response and actual subset rendering. All values below are percent.

| Arm / sign | Response IoU | Response AP | Actual subset IoU |
|---|---:|---:|---:|
| Parent positive | 76.7395 | 92.0912 | 76.1066 |
| Parent signed | 76.7399 | 92.0952 | 76.1453 |
| Frozen expected positive | 76.9234 | 92.2082 | 77.1556 |
| Frozen expected signed | 0.2496 | 81.2336 | 6.2661 |
| Paired legacy positive | 76.7174 | 91.4646 | 75.7609 |
| Paired legacy signed | 76.7245 | 91.4564 | 75.7795 |
| Paired expected positive | 77.4965 | 91.4135 | 77.6696 |
| Paired expected signed | 0.2478 | 78.5729 | 6.4326 |

Expected positive-only output improves IoU while AP falls. Selecting it for
fern would hide the signed failure. Authorized positive-scribble coverage drops
from 99.9478% to 8.8609% after independent negative maxima; the frozen counterpart
falls from 99.9396% to 4.5084%. Negative selected fractions fall to zero. This
reveals over-exclusion using prompt evidence alone, without target-mask access.
It does not isolate whether weak fractional observations, association ambiguity,
learned member errors or their combination dominates.

On one shared-GPU run, expected signed decoding takes about 1.16 seconds versus
0.316 for paired legacy, with peak allocated memory 2.036 versus 1.897 GB.
These are descriptive whole-prediction measurements, not a controlled latency
benchmark. Registration, encoders and construction costs remain separate.

Accepted four-scene actual LERF mean stays 32.3130%, versus saved v10's 66.9633%
under different source/resource conditions. Accepted fern signed response stays
76.7399% versus saved v10's 82.1851%, a 5.4453-point gap under different raw-prompt
and source-access contracts. No new overall gain is claimed.

## All teacher development inputs

These are SAM-region diagnostics, not external image/human 3D annotations.
All default cases, including inherited unsupported zeros, remain in the means.

| Scene / arm | Image subset IoU % | 2D subset IoU % | 3D subset IoU % |
|---|---:|---:|---:|
| Figurines parent | 23.6868 | 37.4463 | 32.2285 |
| Figurines frozen | 23.6868 | 20.5089 | 32.2285 |
| Figurines paired legacy | 26.2219 | 37.8077 | 31.1013 |
| Figurines paired expected | 27.0489 | 29.1838 | 35.2430 |
| Ramen parent | 31.7428 | 40.3042 | 38.8454 |
| Ramen frozen | 31.7428 | 33.2307 | 38.8454 |
| Ramen paired legacy | 31.4468 | 41.0902 | 38.1195 |
| Ramen paired expected | 32.4625 | 37.8297 | 37.7761 |
| Fern parent | 43.4368 | 44.5748 | 44.1197 |
| Fern frozen | 43.4368 | 44.5004 | 44.1197 |
| Fern paired legacy | 44.0609 | 43.4934 | 43.8117 |
| Fern paired expected | 44.2038 | 44.7119 | 44.7675 |

There are 169/149/33 cases per input on figurines/ramen/fern. Figurines retains
one unsupported 3D case. Frozen image/3D means remain exact; the change in 2D
is specific to the new uncertain-support algebra, without isolating its components.

## Source-only nonlinear aggregation check

After observing teacher-development regressions, a read-only audit uses the
unchanged parent models and 32 uniformly spaced training episodes per scene.
It chooses a median interior ray and up to 64 uniform distinct dominant Gaussian
associations per known source sign. These are uncertain associations, not
verified 3D membership labels. Every expected energy is checked against a
weighted sum of explicit singleton energies on the same rows.

Mean response on sampled source-interior associations, equally weighting
episodes rather than rows:

| Scene | Existing prototype | Sigmoid of mean energy | Mean singleton sigmoid |
|---|---:|---:|---:|
| Figurines | 0.5182 | 0.3039 | 0.3817 |
| Ramen | 0.6470 | 0.5082 | 0.5158 |
| Fern | 0.9724 | 0.9553 | 0.9376 |

Energy and response averaging do not commute, and their difference has no
fixed sign here. Switching to response averaging is not established as a remedy:
its interior response remains below the existing prototype in all three scenes,
and its average response on source exclusions increases. These are source
diagnostics, not public IoU, calibration or a new candidate.

The prior resultant-norm audit showed discarded information, but did not
establish that disagreement should directly reduce object membership. This
intervention rejects that simple interpretation. Similarly, idempotent and
monotone negative aggregation does not establish compatible signed evidence.

## Replay, numerical limits and next boundary

Forty-six tests pass: expectation identities, signed monotonicity, duplicate
constraints, gradients, old singleton invariance and local export loading.
All nine models pass copied-field real text/image/2D/3D adapter replay. All six
NVOS jobs replay exactly on the original GPU with original source opens denied.
Enforcement is a Python open hook, not an OS sandbox.

The first replay on another RTX 4090 differed in actual alpha for the paired
expected positive job at 22 pixels, maximum 5.9604645e-8. Members, selected rows,
retained response and the >0.5 mask were exact. The failed attempt, numerical
comparison, same-GPU probe and complete same-GPU repeat are retained. Do not
claim universal cross-device bitwise rendering. Previously established training
non-repeatability remains; these single fits are not significance tests.
No positive promotion exists to trigger the declared training repeats.

Retain the common physical candidate. The next aggregation design must distinguish
uncertain ray association from disagreement about object membership, and combine
positive/negative evidence compatibly rather than turning every uncertain negative
into an equally strong veto. Prompt consistency and evidence-density/resampling
behavior should be checked before target scoring. Neither energy algebra nor
source proxy means establishes a replacement.

Higher semantic responses with worse actual sets do not uniquely diagnose
soft/hard optimization mismatch. Historical
[next67 hard-forward controls](2026-09-30-subset-visibility-support-next67.md)
and [next49 discrete controls](2026-09-29-partial-extent-and-deployment-next49.md)
provided little or negative benefit under their different source-assisted
contracts. Their useful exposed-support correction is already present here.
No ungrounded straight-through, threshold or support-count sweep follows.
Identity/extent evidence and coherent selected geometry remain substantive
requirements; no successful architecture or top-conference novelty is established.
