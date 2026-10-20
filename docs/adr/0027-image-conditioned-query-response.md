# ADR0027: Test image-conditioned query responses and their 3D writeback

Date: 2026-10-17. Status: registered before new target scoring.

The user explicitly prioritizes NVOS SOTA and authorizes reasonable module and
system changes. ADR0024--26's source-only access is a controlled track, not a
permanent restriction on the architecture. Historical scores remain unchanged.

N04 pairs the identical v10 response with either a source-trained RGB render or
the real image at the same camera. Frozen SAM3 receives its >=0.5 bounding extent
and, in the primary branch, continuous response log-odds. A box-only control tests
whether retaining an incomplete dense prior inhibits scope completion. The
official decoder emits one mask; no GT-based candidate selection or task switch.
All images use the same raster and image preprocessing. No new user points,
negative scribbles, manually repaired seed, teacher weights or geometry.

Both dense-prior outputs are also lifted into the original Gaussian response
using confidence-weighted image observations at 512x384 and the existing Huber
solver, initialized from v10, 120 steps. Unsupported elements retain their old
values. Render this actual 3D update with the original v10 domain and readout.
Primary arm is real-RGB dense-prior plus 3D writeback. This is view-conditioned
inference; a target-view image improvement alone does not establish a single
query-global 3D field or multi-view consistency. Source-image agreement and
losses must be reported; no guarantee of preserving source responses is asserted.

Seven fixed arms on all eight tasks; all 56 responses sealed before scoring.
Target RGB is allowed only for the three explicit observed-image arms. No target
GT access before completion. Source-trained RGB parameters must match frozen
geometry and exclude the target view. Check projection against gsplat, identity
replay, label channels, finite scores, resize order and independent confusion/AP.
Report both GT conventions. Other benchmarks remain unchanged references.

External objectives: local LUDVIG 91.2577 is a reproduction target, not proof of
current SOTA. SAGO (arXiv:2607.01628) reports 92.7 NVOS mIoU. VCAR
(arXiv:2608.30870) reports 93.5 but its seven-scene table collapses Horns and
does not establish our exact eight-task prompt/aggregation contract. Keep those
as literature targets pending alignment, not directly verified comparators.

Persist compact source/config/input hashes, metrics, checks and one bounded
diagnostic figure; no new weights or dense predictions. Private tmpfs is capped
at 256 MiB and released after independent scoring.
