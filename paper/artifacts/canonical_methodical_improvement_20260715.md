# Canonical field methodical improvement — 2026-07-15

## Decision rule

This iteration only promotes changes that are query-independent and survive a
held-out, cross-scene feature-fidelity gate.  LERF masks, text labels, ScanNet
instances, NVOS scribbles, and SPIn-NeRF prompts are not used to select any
variant.  A variant that improves one development scene but degrades the frozen
second scene is rejected before downstream benchmark evaluation.

## Retained changes

### Capability-first MPR contract

Official frozen adaptors are applied to every 2D RADIO teacher observation
before multi-view primitive fusion.  This is the mathematically correct target
for contextual, nonlinear adaptors; applying an adaptor after fusing raw RADIO
features is only an approximation.

On Ramen, adaptor-before-MPR and adaptor-after-MPR are observably different for
the official SAM capability (mean cosine 0.978683, p05 0.941509, p01 0.877740).
Training against capability-first primitive targets improves target fidelity:

| Target | Mean cosine | p05 cosine |
|---|---:|---:|
| adaptor after raw MPR | 0.971651 | 0.932943 |
| adaptor before MPR | **0.978762** | **0.951590** |

Held-out rendered SAM fidelity rises from about 0.68958 to 0.69647, and generic
render fine-tuning reaches 0.700217.  Boundary-margin retention remains low
(about 0.038), so this fixes target construction but does not solve contextual
feature compositing by itself.

### Least-squares support initialization

Support completion now has a query-free least-squares initializer.  It encodes
the fused raw target into the field coefficient space and initializes missing
local codes through the pseudoinverse of the frozen fusion projection.  This
raises fallback primitive cosine from 0.4364 to 0.873015 (p05 0.835410) while
preserving primary rows exactly.

This is retained as infrastructure, not enabled in the default field: despite
increasing primitive coverage from 0.57910 to 0.73727, its held-out rendered
fidelity decreases (raw 0.737653 -> 0.726616, DINO 0.828974 -> 0.821904, SAM
0.696469 -> 0.692580).  Representation coverage is therefore not treated as
equivalent to useful visible support.

### Canonical-only compositing audit

The compositing audit can now run directly on the canonical field without a
view-residual checkpoint or uncertainty cache.  This makes the query-free gate
reusable for every scene and prevents optional experimental branches from being
silently required by the evaluator.

## Rejected variants

| Variant | Development observation | Frozen cross-scene / final observation | Decision |
|---|---|---|---|
| Fixed gamma=2 contribution sharpening | Ramen raw 0.737653 -> 0.741180; boundary 0.037447 -> 0.040350 | Figurines raw 0.659581 -> 0.651737; p05 0.409146 -> 0.361165 | reject |
| Adaptive ambiguity sharpening | improved boundary behavior on Ramen | Figurines raw 0.659581 -> 0.652503; p05 0.409146 -> 0.364562 | reject and remove code |
| Extreme-pair boundary-weighted loss | intended to retain teacher boundaries | SAM 0.700217 -> 0.700221; boundary 0.038034 -> 0.038128 | negligible; remove code |
| Reliability-weighted fusion | small SPIn-NeRF gain | NVOS and ScanNet decline | reject |
| Whole-view top-3 MPR | keeps real whole-view vectors | four Ramen held-out frames all decline: 0.7387/0.7629/0.7404/0.7357 -> 0.7174/0.7325/0.7283/0.7030 | reject |
| Learned fallback support completion | primitive coverage and fallback fit improve | raw, DINO, and SAM rendered fidelity all decline | do not enable |

The rejected branches are not part of the default method.  In particular, the
standard alpha compositor and contribution-mean MPR remain the reference until
a stronger query-free alternative passes the cross-scene gate.

## Updated diagnosis

The remaining error is not primarily a benchmark threshold, prompt compiler,
or missing mask post-processing trick.  It is a mismatch between contextual 2D
teacher features and a view-independent primitive field under alpha
compositing.  Primitive target fidelity is already high, while the rendered SAM
boundary margin collapses.  Increasing nominal primitive coverage, selecting a
few views, sharpening contribution weights, or reweighting boundary samples
does not repair that loss and can harm other scenes.

The next method-level iteration should therefore:

1. freeze one explicit observation-lifting contract across datasets (camera,
   depth/alpha visibility, responsibility, per-view normalization, and
   contribution-mean fusion);
2. build capability-first raw/DINO/SAM teachers for multiple scenes and require
   the same held-out fidelity gates everywhere;
3. investigate a minimal canonical-core plus zero-mean view residual only for
   2D rendering, while keeping primitive-domain text, image, and 3D-point query
   interfaces canonical;
4. only after that gate passes, rerun LERF, ScanNet, NVOS, and SPIn-NeRF with
   frozen query compilers and metrics.

This direction is method-level: it does not depend on a scene vocabulary, GT
mask, task-specific threshold, or dataset-specific graph parameter.

## Verification

Focused canonical-field, capability, compositor, completion, and multi-view
teacher-cache tests: **41 passed**.  `git diff --check` passes.
