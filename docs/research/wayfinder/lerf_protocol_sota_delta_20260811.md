# LERF protocol and SOTA delta audit (cutoff: 2026-08-11)

## Resolution

The existing LERF-2D OccamLGS and LERF-3D VALA records pass the repository's
provenance/version/protocol-identity check **for reuse as Validated Protocol
Artifacts (VPAs)**. They should not be rerun merely to chart the wayfinder map.
Their disclosed provenance boundaries remain binding: neither artifact is an
asset-identical, strict-table reproduction of every upstream method component.

No incremental primary-source result examined through the cutoff is already
proven to match the full frozen RADIO-GS Evaluation Contract. Two papers
nevertheless define useful, dated candidate targets:

- **LERF-2D candidate:** SAD-GS arXiv v1, `68.8%` mIoU and `88.7%`
  localization accuracy.
- **LERF-3D candidate:** PairGS arXiv v1 under its raw OpenGaussian-style 3D
  evaluation (no 2D refinement), `60.4%` mIoU, `79.6%` Acc@0.25 and `68.2%`
  Acc@0.50.

Both candidates are **conditional, not frozen SOTA Targets**. SAD-GS does not
publish the held-out camera manifest or decoding constants and its printed
`Overall` mIoU is not the arithmetic mean of its four printed scene rows.
PairGS does not establish exclusion of annotated cameras from semantic lifting,
and publishes Acc@0.25 only as an aggregate. Until those gaps are closed, the
authoritative same-contract floors remain the local VPAs: `63.6200 / 82.8487`
for LERF-2D and `54.1249 / 79.3526 / 56.6114` for LERF-3D.

## Scope and comparison rule

This audit treats the repository's frozen contract as normative rather than
treating every paper table headed “LERF-OVS” as interchangeable. A comparator is
same-contract only if all of the following are identified:

1. the four scenes `figurines`, `ramen`, `teatime`, and `waldo_kitchen`;
2. the released 22 annotated frames and 208 object queries;
3. official query strings, with annotated cameras excluded from semantic
   lifting while RGB geometry may follow the released all-view intent;
4. the correct metric domain (rendered relevance map for LERF-2D, or selected
   Gaussian silhouettes for LERF-3D);
5. per-object/per-scene aggregation followed by an unweighted four-scene macro;
6. frozen query calibration, no GT-selected threshold, no target/source RGB at
   query time, and no benchmark-label-conditioned mapping.

Scores below are percentages. “Delta” is `target - current legal RADIO-GS`, so a
positive number is a deficit.

## VPA provenance and reuse decision

### LERF-2D: OccamLGS

**Facts.** The canonical freeze binds source commit
`eb98bcbedfdeb8770aae51d62c0263bddbc54329`, OpenCLIP
`ViT-B-16 / laion2b_s34b_b88k`, four scenes, 22 annotated frames, 208 queries,
raw-peak selection over three levels, threshold `0.5`, the released `30x30`
activation filter, `7x7` smoothing, and scene-equal aggregation
([freeze, lines 31-65](../../../paper/artifacts/evaluation_protocol_freeze_20260801.yaml#L31-L65)).
Semantic lifting excludes `test.txt`; RGB geometry sees all registered views
([reproduction, lines 3-8](../../../paper/artifacts/occamlgs_lerf2d_strict_reproduction_20260801.md#L3-L8)).
The four result JSONs have recorded SHA-256 hashes, and the scene-equal result is
`63.6200435` mIoU / `82.8487101` localization accuracy
([reproduction, lines 12-18](../../../paper/artifacts/occamlgs_lerf2d_strict_reproduction_20260801.md#L12-L18),
[hashes, lines 29-38](../../../paper/artifacts/occamlgs_lerf2d_strict_reproduction_20260801.md#L29-L38)).

**Fact — provenance boundary.** The registry marks the result diagnostic-only
for a strict paper table because the local checkpoints and input feature bundle
are not an upstream-signed official pretrained release. All six protocol-match
dimensions are nevertheless recorded true
([registry, lines 140-177](../../../paper/artifacts/evaluation_protocol_registry_20260731.yaml#L140-L177)).

**Decision.** Reuse this VPA as the canonical local protocol comparator and
minimum promotion floor. Do not describe it as an official-checkpoint
reproduction. The upstream OccamLGS paper result remains primary-source context
at `61.3 / 82.5` ([BMVC 2025 proceedings](https://bmvc2025.bmva.org/proceedings/694/)).

### LERF-3D: VALA semantic/evaluator path

**Facts.** The canonical freeze binds VALA commit
`48902a541333d65aeb0aebf64ad664777a27c3fc`, the four scenes and 208 queries,
three feature levels, official gsplat marginal contribution, robust lifting
`tau_mass=0.75` / `tau_abs=0.13`, per-query raw-peak level selection, KNN-10,
min-max mapping, fixed threshold `0.6`, selected-only alpha rendering, and
scene-equal aggregation
([freeze, lines 68-101](../../../paper/artifacts/evaluation_protocol_freeze_20260801.yaml#L68-L101)).
The split audit establishes `295/4`, `124/7`, `171/6`, and `182/5`
semantic-train/held-out views and catches the extensionless-stem failure mode
([reproduction, lines 8-33](../../../paper/artifacts/vala_lerf3d_protocol_reproduction_20260801.md#L8-L33)).
The hashed result is `54.1248877` mIoU / `79.3526471` Acc@0.25 /
`56.6114038` Acc@0.50
([registry, lines 341-351](../../../paper/artifacts/evaluation_protocol_registry_20260731.yaml#L341-L351)).

**Fact — provenance boundary.** This is the released VALA semantic pipeline and
evaluator on compatible iteration-30000 OccamLGS RGB geometry, not fresh
paper-identical VALA RGB training
([reproduction, lines 3-6](../../../paper/artifacts/vala_lerf3d_protocol_reproduction_20260801.md#L3-L6)).
The registry therefore marks protocol dimensions true but implementation
identity false
([registry, lines 327-340](../../../paper/artifacts/evaluation_protocol_registry_20260731.yaml#L327-L340)).

**Decision.** Reuse this VPA as the canonical direct-3D evaluator/protocol
comparator and minimum promotion floor. Preserve the compatible-geometry qualifier
in every claim.

## Primary-source delta through 2026-08-11

### LERF-2D: SAD-GS is the strongest numerical candidate, not yet same-contract

**Facts from the primary source.** SAD-GS v1 was submitted on 2026-06-28
([arXiv v1](https://arxiv.org/abs/2606.29376v1)). It renders a 512-dimensional
semantic feature at each pixel, computes text/background relevancy, selects the
highest rendered relevance for localization, and thresholds the rendered map for
segmentation. Its LERF-OVS tables report:

| Metric | Ramen | Teatime | Kitchen | Figurines | Reported overall |
|---|---:|---:|---:|---:|---:|
| mIoU | 77.9 | 72.2 | 68.4 | 57.6 | **68.8** |
| localization accuracy | 85.8 | 89.3 | 96.5 | 83.2 | **88.7** |

The mapping pipeline uses SAM ViT-H, Qwen3-VL object-centric descriptions, a
16-dimensional stored semantic field, 30k RGB iterations and 20k semantic
iterations. The paper defines decoding temperature `T` and foreground threshold
`f_fg`, and reports a sweep on the `ramen` scene, but does not state the chosen
headline values in v1. These details and tables are in the
[primary TeX source](https://arxiv.org/src/2606.29376v1).

**Inference.** This is a LERF-2D-style comparator because the prediction being
scored is the thresholded rendered relevance map, not a silhouette obtained by
first selecting a fixed set of 3D Gaussians.

**Unknown / blocking comparability.** The paper does not identify the frozen 22
camera names, 208-query manifest, or semantic holdout split. Its four printed
mIoU rows average to `69.025`, not the reported `68.8`; that difference cannot be
explained by one-decimal rounding alone. The paper also does not freeze `T` and
`f_fg` in the text. Thus its headline cannot yet be asserted to use the VPA's
aggregation, target-visibility, or calibration contract.

**Material delta.** Current legal RADIO-GS is `51.7373374 / 82.9690780`
([current row, lines 123-137](../../../paper/artifacts/unified_six_task_single_radio_mainline_v2.yaml#L123-L137)).
The deficit to the SAD-GS headline is `+17.0627` mIoU and `+5.7309`
localization points. The deficit to the validated Occam VPA is `+11.8827` mIoU;
RADIO-GS is already `0.1204` localization points above that local floor.

### LERF-3D: PairGS is the closest legal protocol candidate

**Facts from the primary source.** PairGS v1 was submitted on 2026-07-01
([arXiv v1](https://arxiv.org/abs/2607.01140v1)). It explicitly follows the
OpenGaussian 3D evaluation: select 3D clusters from a text query, render the
selected clusters without additional 2D post-processing, and compare the masks
with GT. The four-scene raw-3D table reports `60.4` mIoU and `68.2` Acc@0.50;
the supplement reports `79.6` Acc@0.25. A separate, clearly labelled
method-specific 2D-refinement result is `64.7 / 71.7` mIoU/Acc@0.50 and is not
the proposed target. The paper states that its own hyperparameters are shared
across all scenes and datasets. See the
[primary TeX source](https://arxiv.org/src/2607.01140v1).

**Inference.** The raw result is the closest new literature comparator to the
frozen VALA direct-3D metric domain. It is preferable to mixing refined LaGa or
ReLaGS numbers into the raw selected-Gaussian table. PairGS also exposes why
protocol identity matters: its supplement shows large changes when a method's
own 2D refinement is enabled.

**Unknown / blocking comparability.** PairGS does not state that annotated LERF
cameras are excluded from SAM/CLIP lifting, nor does it publish the 208-query
manifest in the paper. Acc@0.25 appears only as a single aggregate, so its exact
object/scene aggregation is not independently checkable from the table. Its
paper also says thresholds for language-feature baselines were swept to maximize
overall performance; those re-evaluated baseline rows must not replace the
frozen VPA without their selected thresholds and sweep domain.

**Material delta.** Current legal RADIO-GS is `42.3784450 / 75.1771899 /
40.7636981` mIoU/Acc@0.25/Acc@0.50
([current row, lines 186-201](../../../paper/artifacts/unified_six_task_single_radio_mainline_v2.yaml#L186-L201)).
The PairGS candidate deficits are `+18.0216 / +4.4228 / +27.4363` points. The
validated VALA-floor deficits are `+11.7464 / +4.1755 / +15.8477` points.

### Published higher numbers that are not eligible for the frozen target

**GaussDet.** GaussDet v1 was submitted on 2026-06-29
([arXiv v1](https://arxiv.org/abs/2606.30638v1)) and reports `69.9` four-scene
LERF direct-3D mIoU. Its primary source also states that, for LERF-OVS, Qwen-VL
2.5 Instruct is run once per benchmark semantic class on each image; those
detections create the per-instance View-Aggregated Semantic Label Distribution.
It averages three mask levels, filters Gaussians at `0.5`, min-max normalizes and
applies a 3D bilateral filter
([primary TeX source](https://arxiv.org/src/2606.30638v1)).

**Decision.** `69.9` is a useful published ceiling but not a RADIO-GS legal SOTA
Target: it performs benchmark-class-conditioned inference over source images and
persists a label distribution tied to that semantic label space. It also reports
LaGa at `64.1`, while PairGS obtains `52.2` under raw 3D and identifies the
published/refined LaGa result separately. That cross-paper difference is direct
evidence that the GaussDet and PairGS headline protocols are not interchangeable.

**Different-cohort false positives.** OpenVoxel reports `66.2` mIoU on only
three scenes and 42 annotated objects, not the four-scene/208-query contract
([official CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_OpenVoxel_Training-Free_Grouping_and_Captioning_Voxels_for_Open-Vocabulary_3D_Scene_CVPR_2026_paper.pdf)).
NG-GS reports an interactive/mask-supervised three-scene LERF task rather than
the frozen concept-query task
([official CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/html/He_NG-GS_NeRF-guided_3D_Gaussian_Splatting_Segmentation_CVPR_2026_paper.html)).
Neither number belongs in the LERF-2D or LERF-3D SOTA tuple here.

## Dated SOTA Target candidates

These tuples are proposals for `/to-spec`; they must remain `conditional` until
the listed evidence gate passes.

### Candidate `lerf2d_sadgs_v1_20260811`

| Field | Frozen candidate value |
|---|---|
| Cutoff / primary version | 2026-08-11; SAD-GS arXiv `2606.29376v1` (2026-06-28) |
| Task identity | LERF-OVS rendered-view 2D open-vocabulary segmentation and localization |
| Cohort | four canonical scenes; exactly 22 annotated cameras and 208 official queries |
| Target visibility | all-view RGB geometry allowed; annotated cameras excluded from all semantic lifting/teacher inputs |
| Query legality | frozen field + official query string only; no source/target RGB, GT, or query-specific external model |
| Metric / aggregation | per-object mIoU and localization hit, per-scene mean, then equal four-scene macro |
| Candidate thresholds | mIoU `>=68.8%`; localization accuracy `>=88.7%` |
| Tolerance | deterministic score; only one-decimal reporting tolerance (`round(score, 1)` must meet the target), no statistical grace |
| Blocking evidence | exact camera/query manifest, `T`, `f_fg`, evaluator commit, and explanation of 68.8 aggregation |

### Candidate `lerf3d_pairgs_raw_v1_20260811`

| Field | Frozen candidate value |
|---|---|
| Cutoff / primary version | 2026-08-11; PairGS arXiv `2607.01140v1` (2026-07-01) |
| Task identity | LERF-OVS direct 3D open-vocabulary object selection |
| Cohort | four canonical scenes and 208 official queries |
| Target visibility | all-view RGB geometry allowed; annotated cameras excluded from semantic lifting/mask evidence |
| Query legality | frozen field + official query string only; no source/target RGB, GT, query-specific detector, or test-label-conditioned mapping |
| Metric domain | select a fixed Gaussian subset, selected-only render, no target-view 2D refinement |
| Aggregation | per-object metrics, per-scene mean, equal four-scene macro |
| Candidate thresholds | mIoU `>=60.4%`; Acc@0.25 `>=79.6%`; Acc@0.50 `>=68.2%` |
| Tolerance | deterministic score; only one-decimal reporting tolerance (`round(score, 1)` must meet the target), no statistical grace |
| Blocking evidence | held-out camera audit, 208-query manifest, exact Acc@0.25 aggregation, evaluator commit, fixed selector/hierarchy configuration |

## Wayfinder implications

1. Preserve both VPAs and their hashes; no protocol rerun is a prerequisite for
   planning.
2. Do not label either dated candidate “frozen SOTA” until a primary-source
   release or an independent compatibility reproduction closes its blocking
   evidence.
3. Plan LERF-2D work against two gates: first recover the `63.6200` validated
   floor without illegal query evidence, then test the conditional `68.8` target.
4. Plan LERF-3D work against the raw selected-Gaussian protocol: first recover
   the `54.1249 / 79.3526 / 56.6114` validated floor, then test PairGS's
   conditional `60.4 / 79.6 / 68.2` tuple.
5. Keep GaussDet `69.9` as a labelled incompatible ceiling, not a blocker or
   acceptance threshold, unless the project formally changes its query/test-label
   legality boundary.
