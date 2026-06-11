# TPAMI Submission Strategy

## Target Journal

Recommended target: **IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)**.

Rationale:

- The paper is primarily a computer-vision and scene-understanding contribution, not only a graphics rendering paper.
- The main claims match TPAMI scope: vision foundation features, open-vocabulary recognition, 3D scene representation, and cross-dataset semantic querying.
- The work already has a journal-style workload: rendered-view LERF grounding, LERF direct 3D object selection, ScanNet direct point-query transfer, frame-wise foundation-feature comparisons, storage/runtime accounting, qualitative comparison, and failure analysis.

Secondary venues:

- **IJCV**: also suitable, especially if the final paper emphasizes analysis and general 3D understanding over leaderboard-style comparison.
- **ACM TOG**: less ideal unless the paper is rewritten around representation/rendering quality or a graphics-system contribution.
- **TVCG**: reasonable fallback if the final narrative becomes visualization/interactive 3D querying oriented.

Template decision:

- Use IEEE Transactions journal mode via `IEEEtran`.
- Keep the CVPR draft as `paper/radio_gs_draft.tex`.
- Add a TPAMI draft as `paper/radio_gs_tpami.tex`, with local `IEEEtran.cls` and `IEEEtran.bst` vendored under `paper/` for reproducible compilation.

## Core Paper Thesis

**CTF-GS learns a compact foundation-feature scene memory for 3D Gaussian scenes that supports rendered-view open-vocabulary localization, direct 3D primitive selection, and direct point querying.**

The strongest defensible version:

- One compact feature field is trained from frozen RADIO foundation features.
- The field has dual readouts:
  - rendered dense feature maps for LERF-OVS 2D localization;
  - direct compact Gaussian/point features for LERF direct 3D selection and ScanNet direct point-query transfer.
- Multiview Primitive Registration is positioned as an auditable registration bridge and training/diagnostic signal, not as the deployed compact direct-field cache.
- Label-free color-edge component calibration is a classical support calibration step, not an external feature extractor or learned network.
- Official SAM3 box readout remains diagnostic; feature-only SAM3-adaptor boundary readout is the promoted internal boundary readout for rendered-view masks.

Avoid overclaiming:

- Do not claim universal primitive-level SOTA because external baseline rows are published-context rows, not locally rerun same-evaluator rows.
- Do not claim official SAM3 instance segmentation from CTF features; the main internal SAM result is a prompt-conditioned adaptor/boundary readout.
- Do not claim DINO universally improves all secondary metrics; use the selected frozen-head task table and note topology-sensitive caveats.

## How Comparable Papers Frame Their Stories

### LangSplat

Writing pattern:

- Starts from LERF-style language fields being too slow or implicit for fast 3DGS querying.
- Emphasizes 3D language Gaussian representation and efficient rendered language querying.
- Main qualitative figures show rendered-view heatmaps/masks over RGB.

What to borrow:

- Use clear rendered-view OVS visualizations.
- Keep baseline comparison compact.
- Highlight efficiency/storage relative to high-dimensional feature storage.

What not to borrow:

- Do not make rendered-view querying the only claim; CTF-GS now has a stronger dual-readout story.

### OpenGaussian

Writing pattern:

- Explicitly criticizes prior 3DGS open-vocabulary methods for staying at 2D pixel-level parsing.
- Defines point/primitive-level querying as the key missing capability.
- Separates 2D rendered evaluation from direct 3D object selection.

What to borrow:

- Make the 2D-vs-3D protocol distinction explicit in the method and experiment sections.
- Use LERF direct 3D object selection as a central bridge to recent 3DGS understanding papers.
- Report Acc@0.25 and mIoU under query-select-render evaluation.

What to improve:

- CTF-GS should stress compact foundation-feature reconstruction and multiple downstream readouts, not only object-level instance consistency.

### Dr. Splat

Writing pattern:

- Makes a strong contrast between rendering-based feature querying and direct Gaussian-level language embedding registration.
- Uses dominant-ray registration and compact embedding storage as central method elements.
- Evaluates direct 3D localization, selection, and segmentation.

What to borrow:

- Position Multiview Primitive Registration as a principled registration bridge inspired by direct Gaussian assignment.
- Explain why direct Gaussian querying matters for holistic 3D scene understanding.

What to avoid:

- Do not imply CTF-GS uses Dr. Splat's exact dominant-ray protocol unless the raster-contribution registration row is promoted. Current raster-level variants are diagnostics.

### OpenGaFF

Writing pattern:

- Defines a Gaussian Feature Field and evaluates on three aligned tasks: LERF 2D, LERF 3D, and ScanNet.
- Keeps main result tables concise and uses qualitative 2D/3D OVS figures to explain protocol differences.
- Uses codebook/object-consistency language to address fragmentation.

What to borrow:

- Use the same three-task taxonomy:
  1. LERF rendered-view 2D grounding;
  2. LERF direct 3D object selection;
  3. ScanNet direct point-query transfer.
- Keep external baseline rows source-anchored and clearly captioned.
- Add category stability/failure analysis for ScanNet.

What to differentiate:

- CTF-GS is not a codebook-attention field; its novelty is compact foundation-feature reconstruction with multi-protocol readouts and frozen-head downstream usability.

## Recommended TPAMI Structure

1. **Introduction**
   - Motivation: foundation features are strong in 2D but hard to reuse from 3D scenes.
   - Problem: avoid storing raw high-dimensional foundation features or training task-specific classifiers.
   - Method preview: compact foundation-feature scene memory with rendered, primitive, and point readouts.
   - Evidence preview: LERF 2D, LERF direct 3D, ScanNet, frame-wise foundation-feature comparisons, storage/runtime.

2. **Related Work**
   - Language fields and open-vocabulary 3DGS: LERF, LangSplat, LEGaussians.
   - Primitive/instance-level 3DGS understanding: OpenGaussian, Dr. Splat, OpenGaFF, InstanceGaussian, SuperGSeg.
   - Foundation-feature distillation and RADIO/DINO/SAM feature spaces.
   - Compact feature fields and storage-efficient 3D semantics.

3. **Method**
   - Problem setup and foundation-feature target.
   - Contextual Gaussian Feature Field.
   - Foundation-Space Reconstruction and View-Conditioned Feature Calibration.
   - Frozen-Head Geometry Consistency and Multi-Head Foundation Consistency.
   - Direct Gaussian/point readout.
   - Multiview-registration-to-field transfer and Support-Calibrated Primitive Readout.
   - Boundary readouts: internal feature-only SAM3 boundary head versus diagnostic official SAM3 box.

4. **Experiments**
   - Protocol and provenance.
   - LERF rendered-view 2D OVS.
   - LERF direct 3D OVS.
   - ScanNet VALA8 direct point-query.
   - Frame-wise foundation features vs. reconstructed scene field.
   - Ablations and contribution ranking.
   - Storage/runtime.
   - Qualitative results.
   - Failure analysis.

5. **Discussion**
   - Why rendered features can outperform frame-wise foundation features under task readouts.
   - Why Direct3D needs support-calibrated readout, not only feature cosine.
   - Where Multiview Primitive Registration fits as bridge rather than cache.

6. **Limitations**
   - Small/fragmented objects, Waldo Kitchen.
   - External rows are published context.
   - ScanNet is direct point-query transfer, not a full segmentation leaderboard.
   - SAM3 official decoder readout is diagnostic.

## Main Figure and Table Layout

### Main Figures

1. **Fig. 1 Framework**
   - Use `paper/figures/radio_gs_framework.pdf`.
   - Must emphasize one compact field, rendered/primitive/point readouts,
     Multiview Primitive Registration as a training-only registered bridge, and frozen downstream
     heads as supervision/probe interfaces.
   - Current visual audit: `paper/artifacts/figure_quality_audit_tpami_20260531.md`.

2. **Fig. 2 LERF 2D/3D OVS qualitative**
   - Use `paper/figures/lerf_2d3d_ovs_qualitative.png`.
   - Layout: scene/query, GT, reproduced baseline, CTF-GS 2D, CTF-GS 3D.
   - Caption must state 2D query occurs on rendered feature map, while 3D query occurs on Gaussian primitives.

3. **Fig. 3 ScanNet direct point-query qualitative**
   - Use `paper/figures/scannet_openvocab_3d_query_qualitative.png`.
   - Binary query point cloud is better for main paper than full 19-class coloring.

4. **Fig. 4 Direct3D support-calibration ablation**
   - Use `paper/figures/lerf_direct3d_support_policy_ablation_qualitative.png`.
   - If page pressure is high, move this to appendix.

### Main Tables

1. **Table 1 LERF rendered-view OVS**
   - Keep concise: scene rows and macro.
   - External LangSplat/LERF/LEGaussians context table may follow or move to appendix if too much.

2. **Table 2 LERF direct 3D object selection**
   - Include OpenGaussian context, CTF-GS registered multiview analysis, CTF-GS compact, and SAM3 diagnostic.
   - Caption must distinguish compact readout from official SAM3 diagnostic.

3. **Table 3 Direct3D compact-readout ablation**
   - Shows strict one-map, prompt ensemble, color-edge component guard, and score-component guard.
   - This protects the claim boundary.

4. **Table 4 ScanNet VALA8 direct point-query**
   - Use OpenGaFF/VALA published context rows but omit OpenGaFF method row by policy.
   - Report split19/split15/split10 mIoU/mAcc.

5. **Table 5 Frame-wise RADIO vs CTF-GS**
   - Same-evaluator RADIO RGB vs CTF-GS rendered features.
   - This is central for the reconstructed scene-field downstream-usability claim.

6. **Table 6 Quantitative ablation summary**
   - Keep only top contribution rows in main paper.
   - Move full diagnostic/negative rows to appendix.

7. **Table 7 Storage and runtime**
   - Combine storage footprint and efficiency if page budget requires.

### Appendix Tables

- Full LERF boundary calibration.
- ScanNet category stability.
- SAM/DINO task breakdown.
- External baseline provenance.
- Failure cases and per-category audits.
- Negative diagnostics: raster contribution registration, proposal/OPR, prompt/SAM variants that did not promote.

## Immediate Writing Priorities

1. Convert the TPAMI draft from "CVPR freeze report" language to journal language.
2. Expand related work with explicit categories and sharper positioning versus OpenGaFF, OpenGaussian, and Dr. Splat.
3. Merge redundant tables in the main paper; move diagnostics to appendix.
4. Rewrite method section around a clean dual-readout pipeline.
5. Make all claim boundaries explicit in captions, not only in prose.
6. Add a reproducibility/provenance paragraph and artifact manifest reference.

## Current Template Status

- `paper/radio_gs_tpami.tex`: IEEEtran journal-mode TPAMI draft.
- `paper/IEEEtran.cls`: local IEEEtran class file from CTAN.
- `paper/IEEEtran.bst`: local IEEE bibliography style from CTAN.
- `paper/radio_gs_tpami.pdf`: successfully compiled TPAMI-format PDF.
