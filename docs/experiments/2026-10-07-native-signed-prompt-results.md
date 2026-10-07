# Real NVOS scribbles through the common Gaussian field

Branch: `codex/query-preserving-gaussian-field`. The fixed
[protocol](2026-10-07-native-signed-prompt-protocol.md) extends the accepted
physical member field to real registered positive/negative scribbles. No model
or teacher is added to its member decoder, and no source tracker, SAM-derived
query mask, reference RGB, target RGB or query-time fitting is used at inference.
This is one availability-selected, historically opened fern case out of eight,
not a full NVOS result or independent final test.
Sealed inputs, outputs and code receipts are in the
[evidence ledger](2026-10-07-native-signed-prompt-evidence.json).

## Construction and deployment

The designated query-free SAM provider covers 14 authorized fern source frames.
Use 12 training frames `[2,3,4,7,8,9,10,12,13,15,17,19]` and two feature-development
frames `[5,14]`. Target frame 1 is excluded; reference frame 12 is an authorized
query-independent construction view. The geometry was already built from source
images and is held fixed, so development holdout is feature-only.

Fresh native SigLIP masked crops use the existing LERF crop/area recipe; fresh
DINO pixels and source SAM partial regions feed the unchanged observation
compiler. There are 329,038 Gaussians, 294,735 observed rows, 15,801 eligible
inferred anchors and 183 training episodes. The same 32-dimensional, 2000-step
subset-alpha recipe takes 44.84 seconds including its load/diagnostic/export
steps, excluding source construction. The deployment is 1,577,411,680 bytes,
not a compact student. Protected geometry and native tensors remain exact.

The input adapter normalizes each registered ray and pools positive/negative
pixel evidence into existing weighted supports. Area resampling preserves thin
strokes; disjoint original strokes may share coarse cells or Gaussian rows and
are not treated as hard 2D constraints. Ordered CPU float64 pooling avoids
GPU atomic-order differences. The shared group/physical member energy, signed
exclusion rule and threshold are unchanged.

An explicit positive-extent control estimates physical scope from distinct
positive ray centres using the existing source convention of twice spatial
standard deviation, floored by the scene default. It does not include within-ray
depth uncertainty or negative strokes. Here the floor stays active: all arms
use 2.6987667 m. Extent/default outputs are identical, so this experiment supplies
**no evidence of an accuracy benefit from extent inference**.

## Fixed results

All eight combinations of untrained/trained field, positive/signed prompts and
default/extent scale were predicted before target scoring. The four distinct
default-scale rows follow; their extent counterparts have exactly the same
metrics. No scene/input-specific choice of model is made.

| Field | Raw official prompt | Retained response IoU % | Response AP % | Actual subset IoU % |
|---|---|---:|---:|---:|
| Untrained native projection | Positive | 74.9933 | 93.4741 | 77.8366 |
| Untrained native projection | Positive + negative | 74.8546 | 93.5841 | 77.8284 |
| Source fitted physical member | Positive | 76.7395 | 92.0912 | 76.1066 |
| Source fitted physical member | Positive + negative | 76.7399 | 92.0952 | 76.1453 |

The retained response uses full-scene observed-support normalization at 756x1008,
then the historical continuous bilinear resize to the 2988x3985 annotation raster
and >=0.5 threshold. It is distinct from actual selected-Gaussian alpha rendered
at 192x256, bilinearly resized to the declared response and annotation rasters,
and thresholded at >0.5. Neither metric silently substitutes for the other.

Saved v10 fern response IoU/AP are **82.1851/94.7045%**. The fitted signed field's
readout-aligned IoU gap is **5.4453 percentage points**. However, v10 uses a
positive-derived SAM reference mask and source tracks; this field uses raw
scribbles, fewer native-feature source views and no source access at query time.
This numerical gap is not an equal-input/resource improvement or the full-cohort
gap to v10. The four-scene LERF actual subset mean remains 32.3130%; it was not
retrained or improved in this round.

The tradeoff is material. Source fitting raises signed response IoU by 1.8853
points but lowers AP by 1.4889 and actual subset IoU by 1.6831. Actual subset
precision/recall change from 85.6973/89.4470% to 79.2125/95.1609%: better coverage
comes with extra selected background. This is consistent with the existing
partial-supervision false-addition problem, not proof of its unique cause.
Keep the same fitted candidate across tasks; do not replace fern with the
untrained arm to improve a table.

Geometric projection covers all supplied stroke mass. Observed positive mass is
99.7247%, observed negative mass 94.8576%; no nearest-visible fallback is used.
Fitting raises reference positive-stroke coverage from 94.8032% to 99.9396%,
while negative-stroke selected coverage rises from zero to 0.5073%. Signed
exclusion removes only 64 of the fitted positive-only set's 121,929 Gaussians and
does not change that negative-stroke coverage. This motivates checking whether
heterogeneous signed evidence is represented and trained adequately, without
claiming that a mean negative prototype alone explains all errors.

The two held-out source views contain 33 teacher regions. Fitted image/2D/3D
actual subset means are 43.4368/44.5748/44.1197%, with zero unsupported cases.
These remain teacher diagnostics, not real external image or human 3D benchmarks.

## Verification and retained limitations

All eight fresh-process cold replays deny the original output and NVOS dataset
paths, with only copied field/camera/stroke files available. Selected rows,
member scores, actual subset alpha and normalized response are bit-exact. An
intentional original-file open is rejected. This is a Python file-open audit,
not an OS sandbox. The first atomic-pooling attempt was stopped by this check
before scoring and is preserved under `public_atomic_aborted`.

The first scoring attempt found original-resolution annotations, larger than the
prediction raster. After all predictions and cold checks were complete, the
scorer was corrected to apply the existing historical resize before threshold;
no prediction, scale, model or threshold changed. The retained evaluator is
also run independently on the saved fitted-signed and untrained-positive
memberships: response arrays and AP are exact, and IoU differs by at most
2.91e-8 from the retained script's float32 division. Thirty-six distinct
regression tests pass; the seven signed
raster/scorer tests also pass after the resize correction.

The next common-method study should address the recall/false-addition tradeoff
and signed multi-support training with source-only evidence. Require fixed
text/image/registered-point gates and both response and actual subset reporting.
Do not tune fern thresholds, enlarge the source-cache contract at inference,
promote the inactive extent control, or resume failed native-metric weight sweeps.
Full NVOS, ScanNet OVS, UQIS, compactness and independent final testing remain open.
