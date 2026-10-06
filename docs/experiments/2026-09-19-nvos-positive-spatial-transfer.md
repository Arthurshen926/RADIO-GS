# NVOS: preserve positive spatial constraints on the frozen field

This is a bounded query-time control on the previously frozen NVOS image branch,
not a complete four-query method promotion or an independent final test.
It does not change persistent geometry, field weights, native semantic encoding,
source selection, target cameras, or the frozen scoring protocol.

## Question and registered controls

The old positive-only route turns a SAM reference region into an image embedding.
This can discard both the explicitly supplied positive locations and the region's
spatial support. The control asks whether those already-authorized inputs can
correct actual element scores on the same map.

All eight tasks use the same registered settings, without target-dependent policy
selection:

- `baseline`: exact cold replay of the frozen image branch.
- `prompt_only`: optimize temporary element membership scores so the rendered
  response at supported positive scribble pixels reaches 0.55, with a squared
  penalty for departing from baseline element scores.
- `prompt_sam`: additionally use the existing **binary reference SAM region as
  lower-weight positive evidence**. This does not retain SAM continuous logits.
  The complement remains unknown, not negative.

The two interventions use 80 Adam steps, learning rate 0.05, positive squared
hinge loss, SAM positive weight 0.25 and baseline fidelity weight 0.01.
Element scores stay in [0,1], and unobserved elements retain their baseline scores.
Both the fitting and final rendering use the actual frozen gsplat operator; no
2D mask is painted to satisfy prompts. The finite optimizer is not a formal
constraint-satisfaction guarantee. Unsupported reference pixels remain failures
in the all-pixel prompt agreement.

Registration: `output/prompt_transfer_20260919/registration.json`.
Opt-in executable: `scripts/audits/prompt_transfer_20260919_evaluate.py`.
Queue: `scripts/audits/prompt_transfer_20260919_queue.py`.
No frozen default implementation was edited.

## Input and evaluation firewall

Only the reference positive scribble is authorized. Negative scribble and target
RGB reads are denied by a Python audit hook. Training feature/supervision reads
are also denied in the evaluator. The reference RGB, positive scribble, crop,
SAM mask, encoder, geometry and frozen field are hash-bound. The SAM mask and
positive scribble must have exactly the same registered raster dimensions; no
implicit fitting-grid resize is permitted.

All three target predictions are persisted before the target GT becomes readable.
The baseline response must be **bitwise equal** to the previously sealed response
before any scoring begins. Target scoring retains continuous OpenCV bilinear
resizing to the native GT grid, then >=0.5, and the same IoU/AP implementation.
No task is removed and no per-task winning variant is selected.

The existing map has 96-dimensional latent vectors and a 64-dimensional native
semantic prefix. The `measured_support512` path component refers to source support
processing, not a 512-dimensional latent space.

## Validation and interpretation boundaries

The related exact-adjoint and query-conditioned diffusion tests pass (23 tests).
This control uses the gradient of the current renderer rather than reusing a
historical responsibility cache on a different carrier. It does not itself test
graph propagation, update a learned structural feature, or establish that unseen
parts have been correctly completed.

Any gain establishes the usefulness of retaining authorized spatial evidence in
query-time readout. It does not establish that binary SAM positives are always
correct or that positive-only constraints can remove unrelated false positives.

## Complete eight-task results

All baseline responses exactly replay the sealed predictions. Both fixed controls
improve IoU on seven tasks and tie on leaves; none regresses IoU versus baseline.
No per-task policy or winner is substituted.

| Task | Baseline IoU | Positive constraints | Plus SAM positive region |
|---|---:|---:|---:|
| fern | 73.58% | 73.58% | 73.98% |
| flower | 25.76% | 62.89% | 83.03% |
| fortress | 63.62% | 74.16% | 93.96% |
| horns_center | 55.24% | 60.39% | 68.45% |
| horns_left | 59.47% | 59.60% | 60.14% |
| leaves | 23.74% | 23.74% | 23.74% |
| orchids | 42.12% | 55.34% | 82.14% |
| trex | 60.47% | 66.05% | 63.47% |
| **Macro** | **50.50%** | **59.47%** | **68.61%** |

Macro AP is 84.38% → 85.74% → 86.99%. Pixel accuracy is 89.63% → 91.44% → 92.93%.

Minimum per-task IoU improvement is 0 for both controls (leaves). Mean absolute
IoU gains are **8.9674 percentage points** and **18.1116 percentage points**.
All supported positive scribble pixels satisfy the 0.5 threshold in both controls.
Fern retains 435 unsupported pixels out of 22,717: full prompt agreement is
98.085%, not 100%. Other tasks have complete positive support and agreement.

End-to-end three-arm execution took 433.9 seconds for all eight tasks;
per task 48.7–58.4 seconds, mean 54.2 seconds.
These times include cold encoder loading, three field loads, two separate 80-step
optimizations, rendering, persistence and scoring. They do **not** isolate the
net latency of a single 80-step interactive query. This method adds online
temporary-element optimization and is not a pure cosine lookup.

Seal: `paper/artifacts/nvos_positive_spatial_transfer_eight_20260919.json`.

## What improved and what remains

The large flower/fortress/orchids improvements establish a concrete loss of
already-available spatial evidence in the image-vector-only route. This is a
same-map, same-positive-input gain; it does not come from extra target imagery,
negative prompts, different geometry, or a changed score renderer.

The positive-only mechanism cannot solve every scope error. Leaves has zero
element changes in both controls because both the scribble and SAM region were
already sufficiently positive. Its target precision remains 23.78% and recall 99.23%; the residual failure is predominantly over-selection.

There are still side effects: trex SAM-region IoU63.47% is below prompt-only66.05%,
and its AP77.51% is below baseline78.50%. Fern SAM-region AP also falls slightly.
Thus zero IoU regressions versus baseline does not mean every ranking metric or
every additional evidence source is harmless. Reliable scope exclusion and
query-conditioned propagation remain independent unresolved problems.

Retain these as opt-in bounded NVOS prompt candidates. Do not claim a complete
four-query method, independent test, equivalence to signed historical inputs,
or automatic applicability to text queries.
