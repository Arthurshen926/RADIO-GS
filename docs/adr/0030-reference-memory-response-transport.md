# ADR0030: Carry reference identity through cross-view memory before 3D writeback

Date: 2026-10-17. Registered before N07 inference.

N04 establishes useful observed-RGB evidence; N05/N06 reject generic removal of
negative hints and inherited boxes. Earlier next32 reference-mask ensembles
also failed their joint gate. Repeating those prompt scans is not justified.

N07 uses the original v10 reference mask and frozen SAM3 tracker to transfer
reference evidence to the observed target RGB. Compare direct reference/target
two-frame memory with the ordered canonical source-view path between them.
The latter uses only already authorized source frames plus the explicit target
RGB. The static N04 dense-prompt image decoder is freshly replayed as a control.
The same checkpoint and exact image preprocessing are used across these arms;
temporal memory is an established external teacher capability, not our invention.

Each image output is additionally lifted into the original v10 observed Gaussian
domain by the existing 512x384 confidence pooling and 120 Huber updates, then
rendered. Primary candidate: sequence memory plus 3D writeback. Seven arms with
v10; all 56 predictions precede both GT scorings. Fixed 0.5 threshold, no GT
selection, no task-wise switching, no new weights or geometry. Record native
image masks separately from actual 3D re-renderings and source consistency.

This tests functional query transport, not a new feature field. A canonical
camera order is not guaranteed to be a smooth temporal video. Two-frame and
sequence paths coincide when reference and target are adjacent; their equal
output is a useful implementation control. The source mask is fallible, not GT.
Image access is expanded relative to v10 and remains explicitly view-conditioned.
Other four benchmarks stay hash-bound unchanged references. This is development
evidence on an opened cohort. N07 measures binary metrics, not AP.

Use bounded private tmpfs responses; lazy image decoding and offloaded tracker
state avoid persistent video/feature caches. Preserve small source/config/input
receipts, scores and all-eight spatial diagnostics. Failed attempts stay visible.
