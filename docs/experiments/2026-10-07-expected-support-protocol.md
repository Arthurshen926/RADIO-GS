# Fixed expected-support energy study

Starting revision b3ea675. Preserve the physical Gaussian member kernel, native
SigLIP/DINO coordinates, geometry, source masks, input eligibility, physical
scale, thresholds, source budgets and 2000 optimizer updates. No target labels,
query-time source images, extra 3D supervision or benchmark-specific decoder
enter fitting or prediction. This is historically opened development evidence.

## Algebra and inference change

For normalized group features f_s and an uncertain association distribution w,
the common singleton logit is T (f_s(i) dot f_s(j) - b) minus squared physical
distance divided by 2s². Replace a normalized pooled prototype with its expected
singleton energy. The sufficient statistics are the unnormalized mean feature,
mean position and position variance. The distance expectation is distance to
the mean squared plus variance. This is an expected logit, not a calibrated
probability or the expectation of sigmoid responses.

Positive observations of the same target use their weighted mean energy.
Registered raster evidence retains ray identities as transient query data.
Each negative ray first marginalizes its uncertain contributors in energy;
the maximum of the resulting exclusion responses applies to the positive score.
An additional independent negative cannot increase membership; duplicating a
whole constraint does not strengthen it. A flat mean of unrelated backgrounds
is no longer used as one imaginary background prototype. Exact 3D clamping,
text/image hypothesis selection and category-union rules remain unchanged.

All negative rays are evaluated, chunked at 32 for bounded transient memory.
No top-k, ray selection, learned aggregation weights or query-time fitting.
Positive rays are evidence-mass weighted; negative ray max treats every nonzero
supported raster observation as a constraint, including fractional boundary
pixels. Report this sensitivity and cost; do not tune it against target masks.
One-element supports retain the exact old path. Old exports default to the old
normalized-prototype rule. No new per-scene tensors or learned parameters.

## Fixed controls

Run figurines, ramen and fern, with their unchanged 18/18/12 source views:

1. Frozen expected energy: retain all accepted parent tensors; change only
   aggregation. Text/image inferred singleton outputs must remain bit-exact.
2. Paired legacy: from the original initialization, each step renders the
   original single-anchor episode plus one auxiliary episode. Auxiliary forms
   cycle one ray, four positive rays, four positive plus four negative rays.
3. Paired expected energy: same paired episodes and initialization, with the
   new aggregation used in both fitting and deployment.

Both fits average the two losses equally, keep 2000 optimizer updates and the
original seeded episode/first-positive sequence. They have 4000 rendered losses
and 2000 single-anchor presentations, versus 2000/2000 in the accepted parent.
This preserves singleton exposure count, not its gradient weight. Auxiliary
sampling uses a separate RNG and identical draws in both fits. Missing known
source negatives retain an unsigned episode and are reported. Unknown source
pixels never become negative. The expected arm computes separate negative-ray
conditions; compare actual condition counts/time/memory, not equal-compute claims.

Verify the expectation identity (including spatial variance and paired native
metrics), signed monotonicity, duplicate-ray invariance, missing-observation
handling, unchanged singleton behavior, gradients and self-contained export
before launching fits. Run all default teacher image/2D/3D diagnostics, preserving
unsupported cases as zero. Save all six LERF predictions and all six NVOS
positive/signed predictions before opening any new target masks. Complete
copied-field source-denied replay before scoring.

## Gate and stopping rule

Each fitted arm must improve actual text subset IoU in both LERF scenes, and
must not regress any scene's default image/2D/3D teacher mean or fern's positive
and signed response IoU/AP/actual subset IoU relative to the accepted physical
parent. Report expected versus paired legacy separately to isolate aggregation
from extra training. Frozen insertion is diagnostic, not eligible for promotion.

If a fit passes, repeat that fixed fit on the same three scenes to assess
numerical stability, then confirm it unchanged on teatime and waldo before
promoting across LERF. If both fit arms pass, prefer the larger two-scene actual
text mean; ties prefer lower condition count. Do not claim an aggregation gain
if the legacy control explains it. Otherwise keep the parent and all failures.
No support-count, loss-weight, temperature, threshold or per-scene variant sweep.

This is not full NVOS/ScanNet/UQIS coverage, a compact export, a calibrated
uncertainty model or a novelty claim. Any unsupported architectural causal
explanation must remain a hypothesis.
