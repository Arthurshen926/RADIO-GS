# TPAMI Terminology Audit

Date: 2026-06-09

This note audits the current GaussFM terminology from the perspective of a
self-contained top-journal paper. The goal is to make the vocabulary easy to
understand, theoretically grounded, and better aligned with the paper's story:
turning view-local foundation-model features into a compact, queryable 3D scene
memory.

## 1. Recommended Top-Level Vocabulary

The paper should use three levels of terminology:

| Level | Purpose | Recommended style |
| --- | --- | --- |
| Paper thesis | What the work is about | `compact foundation-feature scene memory`, `queryable Gaussian feature field`, `multi-protocol readout` |
| Method modules | What the model does | `contextual Gaussian feature field`, `foundation-space reconstruction`, `view-conditioned calibration`, `primitive support calibration` |
| Implementation/protocol controls | How evidence is produced | `VPR`, `GrabCut`, `SAM3 box`, `prompt ensemble`, `fixed threshold`, `cache` |

The main text should foreground the first two levels. Implementation/protocol
terms should appear only when needed to prevent overclaiming.

## 2. Main Terms To Reconsider

| Current term | Issue | Suggested main-paper term | Where to keep current term |
| --- | --- | --- | --- |
| Historical teacher-centric title | Too training-mechanistic and makes the title sound like distillation rather than a reusable scene representation. | `Compact Foundation-Feature Gaussian Memory` | Avoid in paper-facing titles and summaries. |
| `RADIO reference feature` in title/abstract | Suggests the output is only a copy of RADIO; weakens the stronger claim that multiview reconstruction improves usability. | `foundation feature`, `foundation-feature scene memory` | Keep reference/source wording only in protocol controls. |
| `student` | Invites a narrow teacher-student framing and makes the method sound like a model-compression paper. | `reconstructed scene field`, `rendered field`, `compact field` | Use only if discussing distillation mechanics. |
| `Hybrid Gaussian Code Field` | `Hybrid` is vague; `Code` sounds low-level. The actual idea is per-Gaussian latent memory plus spatial context. | `Contextual Gaussian Feature Field` or `Contextual Gaussian Code Field` | `Hybrid` can remain in implementation artifacts if changing code names is costly. |
| `Foundation-Space Reconstruction` | Clearer than the old teacher-centric wording. | `Foundation-Space Reconstruction` or `Compact Feature Lifting` | Use only after defining that RADIO is the frozen foundation-feature reference. |
| `View-Space Feature Aligner` | Fine, but `aligner` is slightly mechanical. | `View-Conditioned Feature Calibration` | `VFA` can remain as a module acronym if already embedded in tables. |
| `Frozen Geometry-Head Consistency` | Awkward word order; may sound like the geometry head is frozen rather than the consistency target. | `Frozen-Head Geometry Consistency` or `Geometry-Head Consistency` | Keep acronym only after definition. |
| `Frozen Adaptor Consistency` | `Adaptor` is implementation-specific and easy to confuse with official model decoders. | `Frozen-Head Consistency` or `Foundation-Head Consistency` | Use `adaptor` only for RADIO's named adaptor heads. |
| `SAM3-adaptor` | May imply official SAM3 decoder behavior. | `SAM3 adaptor-space probe`, `SAM-compatible frozen-head probe` | Use `official SAM3 decoder` only for true decoder calls. |
| `DINO-CV compact field` | `CV` is ambiguous; readers may parse it as computer vision. | `DINO cross-view consistency field` or `cross-view DINO-consistent field` | Tables can abbreviate after expansion. |
| `View-to-Primitive Registration (VPR)` | Strong concept, but `VPR` is widely used for visual place recognition. | `View-to-Primitive Feature Registration (VPFR)` or `Multiview Primitive Registration (MPR)` | If keeping VPR, expand it every time in main claims and avoid using `VPR` alone in section headings. |
| `support-aware primitive policy` | `Policy` sounds like RL or a heuristic trick; the theory is object-support calibration. | `primitive support calibration`, `support-calibrated primitive readout`, `object-support readout` | Use `policy` only in supplementary implementation detail. |
| `RGB/GrabCut support policy` | Reads like a post-hoc trick; `GrabCut` in the abstract/title-level story weakens novelty. | `label-free color-edge support calibration` | Mention `GrabCut` in limitations, ablations, or supplement. |
| `GT-free` | Efficient but engineering-style. | `label-free`, `annotation-free` | Use `GT-free` in table notes if space constrained. |
| `direct row`, `promoted row`, `main row` | Report/artifact language, not paper language. | `deployed compact readout`, `main compact readout`, `evaluated variant` | Artifact reports. |
| `codebase` references | Breaks the independent paper voice. | Remove or replace with method description. | Implementation docs only. |
| `readout` | Good and useful. | Keep, but define it once as an inference interface from the same compact field. | Everywhere, if used consistently. |

## 3. Suggested Title Options

The current title:

> GaussFM: Compact Foundation-Feature Gaussian Memory for Open-Vocabulary 3D Scene Understanding

is understandable but too teacher-centric. Better top-journal alternatives:

1. **Compact Foundation-Feature Fields for Open-Vocabulary 3D Gaussian Scene Understanding**
   - Best balance of clarity and theory.
   - Keeps the core claim in the title.
   - If keeping acronym `GaussFM`, define it as a legacy method name rather than
     deriving every word from the title.

2. **Queryable Foundation-Feature Gaussian Fields for Open-Vocabulary 3D Scene Understanding**
   - Stronger emphasis on readout/queryability.
   - Slightly less explicit about compactness.

3. **From View Features to Queryable 3D Scene Memory: Compact Foundation Fields for Gaussian Scenes**
   - More journal-like and story-driven.
   - Less conventional for TPAMI because it is longer and less direct.

Recommended title:

> **GaussFM: Compact Foundation-Feature Fields for Open-Vocabulary 3D Gaussian Scene Understanding**

This keeps the existing method brand while moving the title from `teacher copy`
to `foundation-feature scene representation`.

## 4. Suggested Method Vocabulary

Recommended main-paper module names:

| Concept | Recommended name | Rationale |
| --- | --- | --- |
| Stored scene representation | `Contextual Gaussian Feature Field` | Explains per-primitive + spatial-context storage better than `hybrid code`. |
| Decoder from compact codes to RADIO space | `Foundation-Space Reconstruction` | More theoretical than `foundation-space reconstruction`; still precise. |
| View-space correction | `View-Conditioned Feature Calibration` | Connects alignment to view-dependent rendering rather than a black-box refiner. |
| Feature quality and visibility | `Reliability-Visibility Heads` | More interpretable than `quality heads`. |
| Frozen geometry regularizer | `Frozen-Head Geometry Consistency` | Cleaner syntax and better causal meaning. |
| DINO/SAM/SigLIP supervision | `Multi-Head Foundation Consistency` | Better for the multi-head claim than listing adaptors. |
| VPR bridge | `Multiview Primitive Registration` | Avoids conflict with visual place recognition; emphasizes multiview evidence. |
| Direct 3D selector | `Support-Calibrated Primitive Readout` | Makes it sound like an inference interface rather than a heuristic policy. |
| RGB/GrabCut step | `Color-Edge Support Calibration` | Keeps the classical, label-free nature visible without foregrounding GrabCut. |

## 5. Recommended Claim Language

Prefer:

> GaussFM learns a compact foundation-feature scene memory that supports
> complementary readouts: rendered feature maps for novel-view grounding,
> support-calibrated primitive scores for direct 3D object selection, and point
> features for cross-dataset semantic queries.

Avoid:

> GaussFM stores RADIO reference features and uses a support-aware policy to make the
> direct row stronger.

Prefer:

> Registered multiview evidence is used as a training bridge and compressed into
> the deployed compact field.

Avoid:

> VPR is a cache/readout for the main result.

Prefer:

> The direct-3D readout uses label-free color-edge support calibration; no
> learned RGB segmentation model or VPR feature cache is used at inference.

Avoid:

> The row uses RGB/GrabCut, but it is still pure.

## 6. Where The Current Paper Most Needs Wording Cleanup

1. **Title and abstract**
   - Use `GaussFM: Compact Foundation-Feature Gaussian Memory...`.
   - Describe RADIO as a frozen foundation-feature reference, not as the name
     source of the method.

2. **Introduction**
   - Present the central object as `compact foundation-feature scene memory`.
   - Use `teacher-feature reconstruction` as the mechanism, not the motivation.

3. **Method section**
   - Remove `implemented as ... in the codebase` phrasing.
   - Rename subsections toward method concepts:
     - `Hybrid Gaussian Code Field` -> `Contextual Gaussian Feature Field`
     - `Frozen Adaptor Consistency` -> `Multi-Head Foundation Consistency`
     - `Inference Readouts` can remain.

4. **Direct 3D section**
   - Replace `support-aware primitive policy` with
     `support-calibrated primitive readout`.
   - Replace `RGB/GrabCut` in main prose with `label-free color-edge support
     calibration`; put `GrabCut` in a footnote/table note/supplement.

5. **Frame-wise reference comparison**
   - Rename to `Scene Field vs. Frame-Wise Foundation Features`.
   - This better supports the claim that the 3D field can improve downstream
     usability through multiview aggregation.

6. **Limitations**
   - Keep explicit protocol boundaries, but avoid defensive report language.
   - Use `bounded comparison scope`, `label-free support calibration`, and
     `direct point-query probe` instead of `caveat`, `row`, or `not pure`.

## 7. Proposed Final Vocabulary Set

Use these consistently in the polished manuscript:

- **compact foundation-feature scene memory**
- **queryable Gaussian feature field**
- **contextual Gaussian feature field**
- **foundation-space reconstruction**
- **view-conditioned feature calibration**
- **reliability-visibility heads**
- **multi-head foundation consistency**
- **frozen-head geometry consistency**
- **multiview primitive registration**
- **support-calibrated primitive readout**
- **label-free color-edge support calibration**
- **rendered-view readout**
- **direct primitive readout**
- **direct point-query readout**
- **frame-wise foundation features**
- **reconstructed scene field**

Avoid in top-level prose:

- `student`
- `policy`
- `row`
- `promoted`
- `paper-facing`
- `current`
- `codebase`
- `GrabCut` unless in implementation/protocol details
- `VPR` without expansion
- `pure` unless carefully qualified

## 8. Recommended Minimal Migration

If a full rename is too risky close to submission, apply only these high-impact
changes:

1. Change title to **GaussFM: Compact Foundation-Feature Gaussian Memory...**
2. Change `Hybrid Gaussian Code Field` in prose and Figure 1 to
   **Contextual Gaussian Feature Field**.
3. Keep **Foundation-Space Reconstruction** as the decoder-level term.
4. Change `support-aware primitive policy` to
   **support-calibrated primitive readout**.
5. Change `RGB/GrabCut support policy` to
   **label-free color-edge support calibration** in main text.
6. Change old teacher/student comparison wording to
   **Frame-wise Foundation Features vs. Reconstructed Scene Field**.

These edits would make the paper read less like an experiment report and more
like a standalone representation-learning contribution.
