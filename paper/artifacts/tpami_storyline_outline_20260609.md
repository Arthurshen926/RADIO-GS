# TPAMI Storyline and Outline

Date: 2026-06-09

This outline records the paper-facing story after the terminology migration from
teacher-centric wording to foundation-feature scene-memory wording. It is meant
to keep the manuscript, figures, and advisor presentation aligned.

## Central Thesis

CTF-GS learns a compact foundation-feature scene memory for 3D Gaussian scenes.
The stored representation is a queryable Gaussian feature field rather than a
task-specific classifier or a per-view feature cache. The same field supports
three inference interfaces:

1. rendered-view readout for LERF-OVS 2D open-vocabulary grounding;
2. support-calibrated primitive readout for LERF direct 3D object selection;
3. direct point-query readout for ScanNet open-vocabulary point queries.

The key claim is not universal superiority over every external method under all
protocols. The claim is that a compact reconstructed foundation-feature field
can replace frame-wise feature extraction or high-dimensional feature storage in
several open-vocabulary 3D querying protocols, while keeping teachers and
downstream heads frozen.

## Recommended Main-Paper Structure

### 1. Introduction

Goal: motivate why 3D scenes need compact foundation-feature memory.

Story:
- Dense 2D foundation features are strong but view-local.
- Directly storing them per Gaussian is expensive and not necessarily useful.
- A deployed scene should answer 2D novel-view, 3D primitive, and point-query
  tasks from one stored representation.
- CTF-GS reconstructs foundation-space features inside a 3D Gaussian scene and
  exposes multiple readouts.

Main terms:
- compact foundation-feature scene memory
- queryable Gaussian feature field
- multi-protocol readout

### 2. Related Work

Goal: position CTF-GS against language fields, open-vocabulary 3DGS methods,
Gaussian scene representations, and foundation-model regularization.

Story:
- LERF/LangSplat establish rendered open-vocabulary fields.
- OpenGaussian/Dr. Splat/instance-level methods motivate direct primitive or
  instance querying.
- RADIO-style models supply unified dense foundation features.
- CTF-GS differs by reconstructing compact foundation-space features and
  compressing registered multiview evidence into a deployed compact field.

### 3. Method

Goal: explain one representation and its readouts, not a list of disconnected
modules.

Subsections:
- Problem Setup: train from posed RGB and frozen RADIO features; evaluate
  rendered and direct readouts.
- Contextual Gaussian Feature Field: per-Gaussian compact codes plus spatial
  context and reliability/visibility heads.
- Foundation-Space Reconstruction: decode compact codes into RADIO-compatible
  feature space.
- View-Conditioned Feature Calibration: refine rendered features with visibility
  and geometric signals.
- Multi-Head Foundation Consistency: use frozen SigLIP2, DINOv3, SAM3 adaptor
  heads as training constraints without changing inference representation.
- Support-Calibrated Primitive Readout: convert direct primitive scores into
  object support with label-free color-edge and component calibration.
- Multiview Primitive Registration: use registered multiview evidence as a
  training bridge, not as a deployed inference cache.

### 4. Experiments

Goal: separate protocols so readers do not confuse 2D heatmap evaluation,
direct 3D primitive selection, and ScanNet point-query transfer.

Recommended order:
1. Evaluation Protocol and Provenance.
2. LERF-OVS Rendered-View Grounding.
3. LERF Direct 3D Object Selection.
4. Frame-Wise Foundation Features vs. Reconstructed Scene Field.
5. Core Component Ablation.
6. Storage Footprint.
7. Frozen-Head Downstream Probes.
8. ScanNet Direct Point-Query Transfer.
9. Efficiency and Cost.
10. Qualitative Results.
11. Failure Analysis.

### 5. Discussion

Goal: clarify why the representation is useful and bound the claims.

Points:
- Feature reconstruction matters because multiview aggregation can denoise
  frame-wise foundation features.
- Direct primitive querying requires support calibration; primitive scoring and
  object-mask quality are separate issues.
- External baseline numbers are source-anchored context unless locally rerun
  under one evaluator.
- The field can outperform frame-wise RADIO on selected frozen-head tasks, but
  the paper avoids a universal feature-superiority claim.

### 6. Limitations

Goal: state protocol and method boundaries without defensive wording.

Boundaries:
- The method depends on a pretrained Gaussian geometry backbone.
- Small and fragmented objects remain the hardest cases.
- ScanNet is a direct point-query feature probe, not a full segmentation
  leaderboard.
- External baselines are not all same-evaluator reproductions.
- Label-free color-edge support calibration uses RGB edges but no learned RGB
  segmentation network or official RGB SAM decoder.

### 7. Conclusion

Goal: close on the representation-learning contribution.

Suggested close:
CTF-GS turns a 3D Gaussian scene into a compact, reusable foundation-feature
memory. By reconstructing frozen RADIO features and exposing rendered-view,
primitive-level, and point-level readouts, it provides a unified route for
open-vocabulary 3D scene querying with explicit storage, protocol, and failure
analysis.

## Figure and Table Placement

Main figures:
- Figure 1: framework figure with the new terminology.
- Figure 2/3: LERF 2D and 3D open-vocabulary qualitative comparison.
- Figure 4: ScanNet binary open-vocabulary point-query qualitative comparison.
- Figure 5 or appendix: direct-3D support-calibrated readout ablation.

Main tables:
- LERF rendered-view result and compact external context.
- LERF direct 3D local result and compact readout ablation.
- Frame-wise foundation features vs. reconstructed scene field.
- Core component ablation summary.
- Storage footprint.
- ScanNet VALA/OpenGaFF-8 direct point-query context.
- Efficiency and cost.

Appendix tables:
- calibration sweeps;
- multiview-registration controls;
- per-query failure audits;
- additional frozen-head probes;
- additional qualitative examples.

## Terminology Rules

Use in main text:
- compact foundation-feature field;
- compact foundation-feature scene memory;
- contextual Gaussian feature field;
- foundation-space reconstruction;
- view-conditioned feature calibration;
- multi-head foundation consistency;
- multiview primitive registration;
- support-calibrated primitive readout;
- label-free color-edge support calibration;
- frame-wise foundation features;
- reconstructed scene field.

Avoid in top-level prose:
- student;
- direct row;
- promoted row;
- support policy;
- RGB/GrabCut;
- VPR without expansion;
- codebase.

These lower-level terms can remain in file names, artifact names, or
implementation/protocol notes when changing them would reduce traceability.
