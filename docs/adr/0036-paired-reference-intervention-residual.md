# ADR0036: Transfer the paired reference intervention, not the whole teacher

2026-10-18. N13 fixed while N12 predictions are running, BEFORE any N12 target
scores. It is motivated by functional preservation and past whole-response
replacement failures, not selected N12 successes or target-region labels.

Keep exact N12 static target response b, original-reference memory p, and
constrained-reference memory q. The paired memory calls fix target RGB, frozen
model, preprocessing and transport, changing only the reference response.
Primary corrected response is clip(b + q - p, 0, 1). A fixed secondary tests
logit(b) + logit(q) - logit(p), sigmoid readout, probability clipping 1e-4
for numerical stability. Where q == p, explicitly retain b exactly in both.
There are no learned coefficients, target thresholds or task-wise choices.
The delta can add or remove responses; it is not a correctness guarantee.

Evaluate actual native-ray 120-step Huber fitting with original geometry,
initialization, observed domain and abs(2y-1) confidence. Seven arms: v10,
exact parent static_native/constrained_memory_native controls, probability_image/
native (primary), log_odds_image/native. All56 predictions precede target GT.
N12's full replacement is the matched control. Report two GT conventions,
IoU/AP, source residuals and per-task repair/damage. N11 is a previous candidate,
not an additive causal control. Other four benchmarks remain unchanged references.

Capture five parent arrays per task in <=128MiB private tmpfs, hash-bind all
to the complete N12 seal. Start GPU fitting after N12 has sealed all72 arrays;
wait for parent scoring completion before opening N13 GT. This overlaps CPU
scoring with new inference without target leakage. No new model inference.
Use the same explicitly registered 8MiB continuation study, <=1MiB new run
receipts and <=256MiB prediction bank. Release owned arrays after scoring.
Preserve failures and source versions. No default promotion or SOTA claim.
