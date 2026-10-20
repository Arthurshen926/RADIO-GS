# ADR0040: Audit response ranking before interpreting decoder AP loss

2026-10-18. This is a numerical diagnostic of completed N16, not another
candidate or permission to reopen its stopped crop/threshold/decode search.

N16 image IoU is close to N14, but image AP drops by about two points. The
worker converts native SAM logits to float32 sigmoid probabilities. Large
positive logits can collapse to identical ones, so the observed AP drop cannot
yet be assigned entirely to semantic ranking. Inspect all eight tasks, not only
the largest regressions. Replay the frozen N16 worker with read-only observers
at its image-decode and video-propagation interfaces. Require exact hashes of
all 24 quality/full/crop probability maps and unchanged decoder records.

Retain raw logits only in bounded private tmpfs. Seal all tasks before GT access.
Compare, on both existing GT conventions, the original float32-probability
bilinear readout, float64 sigmoid with the SAME bilinear readout, and a stable
ranking of that same probability mixture: independently resize sigmoid(z) and
sigmoid(-z), then use log(p)-log(q). The last score avoids subtracting tiny
complements from one; it is NOT bilinear interpolation of logits. Report native
grid logit/probability AP separately with nearest-resized GT as a diagnostic,
not as benchmark results. Independently verify hard-mask counts and original
image AP. Report saturated TP/FP, ties, numerical response differences, and
any hard-mask changes. No GT affects predictions or chooses a readout.

This audit does not rerun or relabel actual 3D fits. Even if image AP is affected
by precision, N16's observed native 3D AP loss and its original stopping rule
remain in force. No SOTA claim, per-task choice, prompt change or new training.
If quantization does not explain the drop, stop this numerical route. If it
does, correct the causal account and preserve both original and diagnostic
metrics; any subsequent method change needs its own controlled experiment.

Reuse the 8MiB study / 2MiB run / 1MiB file limits, at most 1MiB new durable
records, 256MiB bank and 128MiB per-task scratch. No weights or dense persistent
outputs. Archive sources and external teacher interface hashes; release owned
scratch and bank on success/failure. Existing experiments remain immutable.
