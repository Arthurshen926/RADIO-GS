# TPAMI Storyline and Outline

Date: 2026-06-09

This outline records the paper-facing story after the terminology migration from
teacher-centric wording to foundation-feature scene-memory wording. It is meant
to keep the manuscript, figures, and advisor presentation aligned.

## Central Thesis

CTF-GS learns a compact foundation-feature scene memory for 3D Gaussian scenes.
Its core representation is a compact RADIO Gaussian feature field: low-dimensional
per-Gaussian latent codes are augmented with spatial context, reliability and
visibility cues, and decoded into RADIO-compatible scene features on demand.
The stored representation is a queryable Gaussian feature memory rather than a
task-specific classifier, a per-view feature cache, or a set of separate
DINO/SAM/SigLIP memories. The same memory supports three query modes:

1. rendered-view querying for LERF-OVS 2D open-vocabulary grounding;
2. support-calibrated primitive querying for LERF direct 3D object selection;
3. direct point querying for ScanNet open-vocabulary point queries.

The key claim is that a compact reconstructed foundation-feature field can
replace frame-wise feature extraction or high-dimensional feature storage across
same-protocol reproduced 2D OVS, direct 3D OVS, and ScanNet point-query
benchmarks, while keeping foundation encoders and downstream heads frozen.

## Recommended Main-Paper Structure

### 1. Introduction

Goal: motivate why 3D scenes need compact foundation-feature memory.

Story:
- Dense 2D foundation features are strong but view-local.
- Directly storing them per Gaussian is expensive and not necessarily useful.
- A deployed scene should answer 2D novel-view, 3D primitive, and point-query
  tasks from one stored representation.
- CTF-GS reconstructs RADIO-compatible foundation-space features inside a 3D
  Gaussian scene through a compact contextual feature field.

Main terms:
- compact RADIO Gaussian feature field
- compact foundation-feature scene memory
- queryable Gaussian feature field
- multi-protocol query

### 2. Related Work

Goal: position CTF-GS against language fields, open-vocabulary 3DGS methods,
Gaussian scene representations, and foundation-model regularization.

Story:
- LERF/LangSplat establish rendered open-vocabulary fields.
- OpenGaussian/Dr. Splat/instance-level methods motivate direct primitive or
  instance querying.
- RADIO-style models supply unified dense foundation features.
- CTF-GS differs by learning a compact contextual RADIO feature field and
  compressing registered multiview evidence into the deployed Gaussian memory.

### 3. Method

Goal: explain one representation and its query modes, not a list of disconnected
modules.

Subsections:
- Problem Setup: train from posed RGB and frozen RADIO features; evaluate
  rendered, primitive-level, and point-level queries.
- Compact RADIO Gaussian Feature Field: per-Gaussian compact codes plus spatial
  context, reliability/visibility heads, and HCD/CTR decoding into
  RADIO-compatible features.
- Foundation-Space Reconstruction: decode compact codes into RADIO-compatible
  feature space.
- View-Conditioned Feature Calibration: refine rendered features with visibility
  and geometric signals.
- Dense Reconstruction and Adaptor-Space Regularization: RADIO is the primary
  learned feature; DINOv3/SAM3 structural adaptor losses and SigLIP2 summary
  probes constrain or test the reconstructed RADIO feature without becoming
  separate scene memories.
- Support-Calibrated Primitive Query: convert direct primitive scores into
  object support with label-free color-edge and component calibration.
- Multiview Primitive Registration: use registered multiview evidence as a
  sparse primitive semantic anchor and training bridge, not as a deployed
  inference cache.

### 4. Experiments

Goal: separate protocols so readers do not confuse 2D heatmap evaluation,
direct 3D primitive selection, and ScanNet point-query transfer.

Recommended order:
1. Evaluation Protocol and Provenance.
2. LERF-OVS Rendered-View Grounding.
3. LERF Direct 3D Object Selection.
4. Frame-Wise RADIO Features vs. Reconstructed Scene Field.
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
- LERF 2D OVS, LERF 3D OVS, and ScanNet comparisons are organized as
  same-protocol reproduced benchmark tables.
- The field outperforms frame-wise RADIO on the selected primary frozen-head
  usability metrics reported in the paper; secondary probes and failure cases
  remain in the appendix.

### 6. Limitations

Goal: state protocol and method boundaries without defensive wording.

Boundaries:
- The method depends on a pretrained Gaussian geometry backbone.
- Small and fragmented objects remain the hardest cases.
- ScanNet is a direct point-query feature probe, not a full segmentation
  leaderboard.
- Historical provenance notes are kept in the appendix only; the main tables
  use the same reproduced protocols reported in the paper.
- Label-free color-edge support calibration uses RGB edges but no learned RGB
  segmentation network or official RGB SAM decoder.

### 7. Conclusion

Goal: close on the representation-learning contribution.

Suggested close:
CTF-GS turns a 3D Gaussian scene into a compact, reusable foundation-feature
memory. By reconstructing frozen RADIO features and supporting rendered-view,
primitive-level, and point-level queries, it provides a unified route for
open-vocabulary 3D scene understanding with explicit storage, protocol, and
failure analysis.

## Figure and Table Placement

Main figures:
- Figure 1: framework figure with the new terminology.
- Figure 2/3: LERF 2D and 3D open-vocabulary qualitative comparison.
- Figure 4: ScanNet binary open-vocabulary point-query qualitative comparison.
- Figure 5 or appendix: direct-3D support-calibrated selection ablation.

Main tables:
- LERF 2D/3D OVS same-protocol quantitative comparison.
- LERF direct 3D compact query/support-calibration ablation.
- Frame-wise RADIO features vs. reconstructed scene field.
- Core component ablation summary.
- Storage footprint.
- ScanNet VALA/OpenGaFF-8 same-protocol direct point-query comparison.
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
- compact RADIO Gaussian feature field;
- compact foundation-feature scene memory;
- contextual Gaussian feature field;
- foundation-space reconstruction;
- view-conditioned feature calibration;
- dense reconstruction and adaptor-space regularization;
- multiview primitive registration;
- support-calibrated primitive query;
- label-free color-edge support calibration;
- frame-wise foundation features;
- reconstructed scene field.

Avoid in top-level prose:
- student;
- direct row;
- promoted row;
- support policy;
- readout as the main conceptual term;
- RGB/GrabCut;
- VPR without expansion;
- codebase.

These lower-level terms can remain in file names, artifact names, or
implementation/protocol notes when changing them would reduce traceability.
