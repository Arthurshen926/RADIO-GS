# ADR0031: Test resolution-consistent response writeback

Date: 2026-10-17. Registered during N07 prediction, before N07 target scoring.

N04 loses 0.7435 IoU points between its direct image evidence and actual 3D
re-rendering. N08 isolates one transport mismatch: evidence pooled on 512x384
rays is fitted against a renderer sampled on those rays, whereas final output
uses 1008x756 rays. Pooling images and sampling a volumetric renderer need not
commute. This is a capacity/transport control, not a claimed new renderer.

Use the exact sealed N07 image probabilities for all three evidence providers:
static dense prompting, two-frame reference memory, and source-sequence memory.
Reuse the N07 coarse writebacks and v10 as frozen controls. New writebacks fit
native 1008x756 rays with the same probability confidence abs(2p-1), original
observed Gaussian domain, v10 initialization, 120 Huber steps and 0.5 readout.
No new teacher calls, no geometry changes, no target-GT parameter choices.
Primary candidate is sequence-memory native writeback; no per-task selection.

All eight tasks and seven arms must be sealed before N08 scoring. Keep both GT
tracks and independent integer confusion counts. Retain all-source teacher
consistency checks. AP is not measured. View-conditioned input scope and the
other four benchmarks' unchanged-reference status remain explicit.

Temporarily hard-link at most 256 MiB of the exact N07 response bank in private
RAM; wait for the full parent seal before using it. Bind every array digest,
parent source snapshot and input hashes. Release these temporary arrays after
scoring. Save only source/config receipts, metrics and a small all-task error
figure. A missing/incomplete parent fails without reading GT.
