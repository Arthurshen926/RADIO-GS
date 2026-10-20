# ADR0033: Separate semantic evidence from a second image segmentation

2026-10-17. Registered while N09 predictions are in progress, before its scores.

N09 compares scribble-only authority with pseudo-mask authority, followed by
SAM refinement. A complete response architecture must also test whether that
decoder removes useful scope from the frozen affinity response. N10 fits the
exact two N09 affinity probabilities directly to native image rays, with the
same confidence abs(2p-1), v10 initialization/observed domain, Gaussian geometry,
120 Huber steps, optional gsplat-compatible footprint and 0.5 readout.

Six arms: exact parent v10/static_native/signed_native/pseudo_native controls;
signed_direct_native (primary) and pseudo_direct_native. All 48 predictions
precede target GT. Both GT tracks and AP; independent binary confusion counts.
Parent teacher arrays, input/config/source hashes and controls must bind exactly.
No new teacher call, user prompt, target-driven scale or per-task selection.
This is a view-conditioned development intervention. Other tasks stay references.

Capture only the six required parent arrays per task via private bounded RAM,
wait for the complete parent seal and release of its bank before GPU work.
Keep N09's archived dependency versions by hash reference, snapshot only new
runtime sources. At most 512KiB new durable receipts, within the same immutable
2GiB study budget. No new weight/dense durable artifact. Preserve failures.

Execution amendment, before any N10 prediction: N09 has sealed all72 arrays
and now performs CPU scoring. The waiting N10 v1 was stopped with zero
predictions/GT reads and its private arrays released. N10 v2 starts fitting
after the parent prediction seal and waits for complete parent scoring before
its own GT access. This overlaps independent compute without changing evidence,
methods or readouts. Keep v1 receipts and the original scheduling decision.
