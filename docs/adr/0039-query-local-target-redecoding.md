# ADR0039: Test target re-decoding after complete query transport

2026-10-18. N16 is fixed after N15 development results and before prediction.
N15 equal-group multiview fusion remains stopped; no fusion-weight/view sweep.

Hypothesis: scope completion and reference transport recover identity/extent,
but the transported target response still has interior FP and thin-structure
errors. Test whether a fresh target image decode of this COMPLETE response can
repair those errors, and whether object-relative image scale contributes.

Replay N14 source-quality selection and its image response exactly. Full-frame
control: frozen SAM3 takes the complete response's same signed log-odds and
half-open box through the established response_prompt interface. Local primary:
crop the native target RGB and that same response to the >=0.5 predicted extent,
expanded by 25% of its width/height on each side and clipped to the image. Both
RGB and dense prior use identical crop coordinates. Decode once, with the same
teacher/resolution and prompt interface, then paste the local probabilities
back at their original resolution. Preserve EVERY probability outside the crop;
an empty response falls back unchanged. No extra points, teacher, training,
target annotation, oracle quality selection, recursive correction or blending.
The fixed context fraction is a design choice, not a tuned optimum.

Seven arms: v10, quality_image/native, full_image/native, crop_image/native.
Primary crop_native. Native fitting uses original geometry, observed domain,
initial response, confidence |2p-1|, 120 Huber steps and threshold 0.5. Image
and actual 3D results are distinct. All56 predictions precede both GT tracks.
Check parent image hashes/native confusion, crop alignment/outside preservation,
projection arithmetic and independent integer confusion. Report AP, P/R, source
teacher residuals and all eight tasks. Other23 jobs remain verified references.

Stop this refinement branch if neither fixed native candidate improves mean
IoU without reducing AP on BOTH GT conventions versus replayed N14. Do not
retune context size or combine per-task winners after scoring. This is adaptive
development, not an independent final test or proof of a universal upper bound.

Use the existing 8MiB study / 2MiB run / 1MiB file limits, <=1MiB new run,
256MiB volatile prediction bank and 128MiB plot links. No new weights or durable
dense arrays; preserve failed attempts and release owned transient directories.
