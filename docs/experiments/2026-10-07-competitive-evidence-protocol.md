# Fixed relative signed-evidence study

Starting revision a1a3b8e. Keep the accepted physical fields, every learned
tensor/native coordinate, geometry, scale, candidates and thresholds. This is
a frozen common-member inference study, with no refit or extra teacher.

The old product p(positive)*(1-p(negative)) suppresses a strong positive even
when its energy exceeds the negative. Test relative competition through the
same member logit: exp(l+) / (1 + exp(l+) + exp(l-)). Implement it stably as
sigmoid(l+ - softplus(l-)), retaining original unsaturated positive logits.
The null score remains 1. These learned scores are not calibrated probabilities.

Two fixed controls isolate the change:

* competitive_pooled keeps the original pooled negative prototype.
* competitive_rays keeps every negative ray and its observed evidence mass;
  each ray uses the accepted normalized prototype/centre kernel. Negative
  evidence is a mass-normalized mixture of exponentiated ray logits, accumulated
  by log-sum-exp in fixed chunks of 32. No expected-energy variance penalty,
  independent maximum veto, per-query fit, ray top-k or inference source access.

The positive adapter remains exactly the accepted pooled adapter. Splitting
identical ray evidence or repeating the entire evidence measure preserves the
mixture algebra; actual raster changes can still change geometric associations.
No negatives, or no observed negative mass, must return the exact old member
scores. Authorized exact 3D constraints remain query-only. The rule accepts
typed evidence in the shared kernel; it contains no dataset/modality routing.
All new policies remain opt-in until sufficient evidence supports adoption.

## Mechanism checks and development cohort

Before prediction, verify stable competition, confidence saturation, gradients,
mass-weighted mixture identity, split-ray invariance, exact unsigned four-input
behavior and self-contained serialization. Keep all preceding regression tests.
Create frozen exports for figurines, ramen and fern from the accepted parents.
No scene tensors or learned parameters may change.

On each scene's two held-out source views, evaluate every original SAM region
with its existing centroid-interior positive ray and up to four uniformly spaced
distinct observed negative pixels from the original known comparable-scope
exclusions. Use default scale, retain unsupported zeros, and compare product,
pooled competition and ray competition. This is a signed teacher diagnostic,
not human 3D annotation or source-mask truth. No missing negative is invented.
Report normalized response and actual selected-subset IoU separately.

For fern, persist six raw official prompt predictions (three policies times
positive/signed) before target scoring. Parent outputs must replay the saved
parent. Both competitive positive-only outputs must be exact. Inspect authorized
prompt agreement first: a candidate loses the prompt gate if signed positive
coverage drops by more than one percentage point from parent or negative selected
coverage increases. This is a predeclared acceptance check, not a runtime
fallback or tuning parameter. All fixed arms remain in the scored report.

Require copied-field source-denied replay on the original prediction GPU. Report
cross-device alpha limits already observed; never claim universal bitwise CUDA
rendering. Replay real text/image/2D/3D adapters for all six new exports, and
verify unchanged full LERF text outputs on the accepted four-scene cohort.

## Gate and scope

A candidate must pass prompt agreement, preserve each scene's signed teacher
actual-subset mean, and improve fern signed response IoU without lowering AP or
actual-subset IoU. Its positive-only/unsigned outputs must remain exact. If both
pass, prefer the larger fern signed response IoU; ties prefer pooled computation.
Report all failures and both policies, with no per-scene/input selection.

A one-case gain does not authorize full NVOS promotion. If an arm passes, use the
unchanged arm on a second NVOS task chosen only from construction-input
availability, with no new coefficients, thresholds or source-query resources.
If neither passes, retain the physical parent; no weight/count/threshold sweep.
Do not relabel a positive-only improvement as signed capability or claim LERF
improvement when unsigned behavior is unchanged. This is historically opened
development, not independent final testing, full five-task success or compactness.
