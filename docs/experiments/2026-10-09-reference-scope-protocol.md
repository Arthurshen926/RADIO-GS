# Current-reference input bridge and field-owned scope composition

Complete eight NVOS requests, seven frozen fields, same old120 source frames.
No model training, extra scene state, runtime source RGB/masks/tracks, old response
fitter, target-camera-conditioned decoding, target labels or per-case routing.
Explicit exception to raw-scribble input: two arms accept the EXISTING v10 positive-
derived SAM reference query mask. This is a current query input, not an extra
construction view. Inherit its RGB/SAM production cost; cached generation is not
rerun and is NOT claimed migrated. Validate mask hash, positive-stroke hash,
producer receipt, frame, raster and not-GT declarations before prediction.

Four fixed arms:
1. raw_signed: existing positive/negative expected-kernel competition baseline.
2. raw_positive: same positive expected kernel, omit negative stroke. Sign control.
3. mask_rays: positive-only expanded reference mask, area resampling, every ray
   and contributor retained by the same expected-energy union; no top-k sampling.
4. Primary mask_composition: same expanded reference mask; project ALL field-owned
   Bpositive>0 regional Gaussian subsets (intersect member_available) into the
   REFERENCE camera with actual subset transmittance. Threshold candidate alpha
   at0.5. Greedily add the region strictly improving fractional reference-mask
   soft-IoU of the 2D union; deterministic lowest-index ties, no tuned part count.
   Union the selected Gaussian subsets; its binary membership is the output.
   Independently render its actual reference and target selected subsets.

raw_positive->mask_rays isolates reference expansion at identical decoder/sign.
mask_rays->mask_composition compares decoders at IDENTICAL expanded query input.
raw_signed->primary is a mixed-contract comparison, not standard raw-input gain.
Composition is an alternative conditional member decoder within the same field,
NOT the unchanged local member kernel. It uses binary scores, not calibrated
probabilities. Greedy matching is not novel by itself, nor globally optimal.
The current reference mask is valid query evidence; target GT is never used to
choose parts. No outputs from prior GT-assisted diagnostics may be read.

All32 predictions and eight exact-selection baseline replays must precede scoring.
Score formal actual-set and retained response metrics, all cases, precision/recall.
Record reference-mask fit and original-stroke agreement separately. Do not turn
reference-fit quality into target-scored routing. Compare v10 only on response
metric and declare source/input construction differences. No default promotion;
common full-five validation and a raw-query composition path remain outstanding.
