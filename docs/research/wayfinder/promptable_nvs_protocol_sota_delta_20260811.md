# Promptable NVS protocol and SOTA delta audit

**Cut-off:** 2026-08-11

**Issue:** [#3](https://github.com/Arthurshen926/RADIO-GS/issues/3)

**Status:** research complete; some target rows remain intentionally unresolved

## Question and answer

This audit asks which published numbers can be used as SOTA Targets for RADIO-GS on
NVOS and SPIn-NeRF without collapsing distinct prompt, visibility, scene-cohort, or
query-time-compute protocols.

The short answer is:

1. **NVOS strict-unseen:** the original NVOS 8-task result, 70.1% foreground
   IoU, is the only dated, primary-source comparator found whose stated unseen-view
   setup is plausibly compatible. RADIO-GS's frozen 74.777124% result is +4.677124
   percentage points above it. This is a *provisional same-protocol candidate*, not
   yet a defensible SOTA claim, because exact image/camera-manifest identity has not
   been reproduced locally.
2. **NVOS online-multiview:** keep this as a separate diagnostic track. LUDVIG-DINOv2
   at 92.4% is the dated same-paper target for LUDVIG's all-view protocol; the exact
   local LUDVIG-SAM reproduction is 91.257685%. WildSeg3D reports a higher 94.1%,
   but its random prompt and five-view sampling are not fully specified, so 94.1%
   is broad online context rather than a frozen same-protocol target.
3. **SPIn-NeRF sparse-point:** no numeric SOTA Target can yet be registered. SAGA,
   OmniSeg3D, WildSeg3D, SAGOnline, and MV-SAM do not jointly disclose the point
   budget, sampler, seed, and exact ten-scene cohort required for exact comparison.
4. **SPIn-NeRF full-reference-mask:** LUDVIG's 93.8% published ten-scene result is
   the provisional dated target for its all-view/full-mask protocol. The validated
   local reproduction is only nine scenes (Fork is missing), so a local full-ten
   delta cannot be computed.

Higher headline values from incompatible tracks must not replace any of these
targets. In particular, using target/all-view RGB, online SAM/SAM2, a mask cache,
random ground-truth-derived clicks, or a reduced scene cohort changes the evaluated
capability.

## Evidence policy

The repository's existing protocol audit, registry, locks, immutable result hashes,
and validator outputs are treated as **Validated Protocol Artifacts**. They were
checked first and are reused where their identity remains intact. External changes
after those artifacts were frozen were checked only against primary sources:
official papers, official proceedings, arXiv version records, and official code
where necessary.

Labels used below:

- **Fact**: directly stated by a primary source or encoded in a validated artifact.
- **Inference**: a comparison or eligibility conclusion derived from facts.
- **Unknown**: evidence needed for exact protocol identity was not disclosed or not
  present locally.

"Same protocol" means the prompt, prompt-generation budget and seed, scene cohort,
reference/evaluation split, query-visible inputs, query-time external models,
metric, and aggregation all match. Similar dataset names and matching metric labels
are insufficient.

## Validated artifact provenance check

The authoritative repository artifacts are:

- `paper/artifacts/promptable_nvs_protocol_registry.yaml`
- `paper/artifacts/evaluation_protocol_freeze_20260801.yaml`
- `paper/artifacts/evaluation_protocol_registry_20260731.yaml`
- `paper/artifacts/ludvig_nvos_spin_protocol_audit_20260731.md`
- `reproductions/ludvig/upstream.lock.json`
- `reproductions/ludvig/official_3dgs.lock.json`
- `reproductions/ludvig/patches/0001-reproduction-seeds-and-json-results.patch`

| Check | Finding | Classification |
| --- | --- | --- |
| Protocol validators | The evaluation validators pass: 7 frozen protocols and 27 registry rows. | **Fact** |
| LUDVIG upstream | Pinned to `naver/ludvig` commit `4461fc515439bb498a75d71738a1e73cf7a452ed`; Segment Anything is pinned to `6fdee8f2727f4506cfbbe553e23b895e27956588`; the audited official 3DGS revision is `f7a116fb1397d9842239127d39dc212f93171f70`. | **Fact** |
| Reproduction patch | SHA-256 `2c21257316c6f65d25eea2bbd98481bd3e42f0d84df23a13c1bd1cb645e7d602` matches the freeze. | **Fact** |
| NVOS immutable summary | `output/protocol_audit_20260731/ludvig/nvos/released_all_view_full8_3seed_summary.json`, SHA-256 `65e1f8e5c1f17083f66e5b7d4f6f03687f6806394c78a5cfce25546ca42e3546`, freezes 91.25768502741802% macro IoU. | **Fact** |
| SPIn immutable summary | The nine-scene summary, SHA-256 `ee300d2eb805600374461f953eb7a89ad1c890c2f02bbb347957e2f164e75e17`, freezes 93.7200449592385% macro IoU. | **Fact** |
| Launcher binding | The current `reproductions/ludvig/run_ludvig_sam.py` hashes to `81714a8942c4359261909e3a90a1f45eefc80d69cf91b15a2e12756478745e25`; the registry row freezes `1f1bc95d70f22fcfb1cfe2df1bc3f2416ebb667c2faeddb7dc24bcd9bceed68f`. The change after the frozen version added a physical-GPU selector and optional materialization/carrier-retention controls. | **Fact** |
| Effect of launcher drift | The frozen result files remain immutable and their manifests bind the historical reproduction. The current launcher appears default-protocol-equivalent, but byte identity is not proven. Reuse the frozen results; do not claim that a new run from the current launcher is the same artifact until a historical checkout or an explicit equivalence binding is provided. | **Inference / Unknown** |

The validated audit also recomputes like-for-like paper subsets rather than comparing
rounded headlines: local NVOS-SAM is 91.257685% versus 91.3375% for the paper's same
eight tasks (-0.079815 points), while local SPIn-SAM is 93.720045% versus 94.577778%
for the paper's same nine available scenes (-0.857733 points). Neither difference
changes protocol identity. The latter must not be compared directly with the rounded
93.8% published full-ten headline because Fork changes the cohort.

The launcher mismatch is one failing provenance assertion among 84 focused
reproduction/registry tests (83 pass, 1 fail). It does not invalidate the already
hashed summaries, but it does prevent a blanket statement that the present worktree
is an exact rerun environment.

## Frozen protocol identities

### NVOS strict-unseen (`nvos_strict_unseen_v1`)

**Fact.** The frozen cohort comprises eight tasks: fern, flower, fortress,
horns-center, horns-left, leaves, orchids, and trex. Each task uses the official
positive and negative scribbles on a reference image and scores one different target
image. The reference image is not scored. Foreground IoU and pixel accuracy are
computed for the target, then macro-averaged equally over tasks. The target RGB is
forbidden during both field construction and query; the target mask is scoring-only.

**Unknown.** The repository has not yet bound the original NVOS release's exact
camera/image manifest to the frozen task manifests byte-for-byte. This is the last
identity check between the original paper's unseen-view row and
`nvos_strict_unseen_v1`.

### NVOS online-multiview (`ludvig_official_online_multiview_v1`)

**Fact.** LUDVIG trains its 3DGS on all scene views for 30,000 iterations, including
the eventual target images, and performs prompt-specific uplifting by invoking a 2D
foundation model on target/all-view RGB. For NVOS-SAM the local reproduction uses
the positive scribble, draws three positive points per SAM call, runs ten calls per
view, min-max normalizes, and applies the fixed threshold. Negative scribbles are not
used. Target masks remain scoring-only. The paper reports the mean over three runs;
the release exposes only seed 0 and the validated patch exposes seeds 0, 1, and 2.

**Inference.** This is a valid online-multiview benchmark but is categorically not
NVOS strict-unseen. It also violates RADIO-GS's proposed destination if used as the
primary query path because query-time target RGB and an external 2D model are visible.

### SPIn-NeRF sparse-point (`spin_nerf_point_prompt_10scene_v1`)

**Fact.** The intended cohort is orchids, leaves, fern, room, horns, fortress, Fork,
pinecone, truck, and lego. SPIn-NeRF describes sparse positive/negative user points
in one source view; SAGA's supplement generates a random subset of points inside
and outside the reference mask. The reference view is prompt input and is not scored;
target masks are scoring-only.

**Unknown.** SAGA does not disclose the random subset size or seed. The repository
therefore correctly marks this row `pending_prompt_sampler_freeze`. No result using
an independently chosen point budget is an exact comparator.

### SPIn-NeRF full-reference-mask

**Fact.** `spin_nerf_full_reference_mask_10scene_v1` passes the complete first
annotated reference mask as the prompt. It uses the same ten scenes and macro-averages
per-frame IoU within each scene, then scenes equally. All RGB and poses may be used
during field training; the reference mask is legal task input; target masks are
scoring-only. This is a full-mask support diagnostic, not the sparse-point canonical
task.

**Fact.** LUDVIG additionally selects a SAM candidate and threshold per scene on the
reference ground-truth mask, then scores only target views. The published full-ten
macro is 93.8%. The exact local three-seed reproduction contains only nine scenes:
Fork is absent. Its 93.7200449592385% must not be labeled a ten-scene result.

## Primary-source increment since the validated artifacts

The table records material published context found by the 2026-08-11 cut-off. A
number is included only to explain why it is or is not eligible; inclusion is not an
endorsement of protocol identity.

| Method and primary source | Version / result | Prompt, visibility, and cohort facts | Eligibility conclusion |
| --- | --- | --- | --- |
| [Neural Volumetric Object Selection](https://openaccess.thecvf.com/content/CVPR2022/html/Ren_Neural_Volumetric_Object_Selection_CVPR_2022_paper.html) | CVPR 2022; 70.1 NVOS foreground IoU, 92.0 accuracy | Official foreground/background scribbles in one reference image; one different unseen validation image per scene; eight LLFF tasks. | **Provisional strict-unseen comparator.** Exact release-manifest identity remains unknown. |
| [SPIn-NeRF](https://arxiv.org/abs/2211.12254) | CVPR 2023; 90.96 segmentation IoU (91.66 for the two-stage variant) | Sparse points in one source view create a source mask; video segmentation propagates it; the dataset contains ten scenes. The paper does not freeze a downstream point-sampling seed for later feature-field comparisons. | **Context only** for the sparse-point track until the prompt generator and exact evaluation cohort are bound. |
| [SAGA](https://ojs.aaai.org/index.php/AAAI/article/view/32193) | AAAI 2025 final; 92.6 NVOS, 93.4 SPIn | Random positive/negative prompt points are sampled from NVOS scribbles or inside/outside the SPIn reference mask; count and seed are undisclosed. Target-RGB exclusion is not verifiable from the released evaluator. | Not exact strict-unseen or deterministic sparse-point. Use final values, not older project-PDF values 90.9/88.0. |
| [OmniSeg3D](https://openaccess.thecvf.com/content/CVPR2024/papers/Ying_OmniSeg3D_Omniversal_3D_Segmentation_via_Hierarchical_Contrastive_Learning_CVPR_2024_paper.pdf) | CVPR 2024 final; 91.7 NVOS, 94.3 SPIn | Reusable 3D feature representation, but the published prompt sampling is not identical to either frozen official scribbles or the full-reference-mask LUDVIG diagnostic. | Context only. Use final 94.3, not the older 95.2 preprint value later copied into some tables. |
| [LUDVIG](https://openaccess.thecvf.com/content/ICCV2025/papers/Marrie_LUDVIG_Learning-Free_Uplifting_of_2D_Visual_Features_to_Gaussian_Splatting_ICCV_2025_paper.pdf) | ICCV 2025; NVOS SAM 91.3, DINOv2 92.4, SAM2 91.3; SPIn 93.8 | All-view 3DGS and query-specific 2D-feature uplifting; NVOS includes target RGB; SPIn uses a full reference mask and reference-GT candidate/threshold selection. | Exact online family; never merge with strict-unseen or sparse-point. Local exact reproduction exists for NVOS full-eight and SPIn nine-of-ten. |
| [WildSeg3D](https://openaccess.thecvf.com/content/ICCV2025/html/Guo_WildSeg3D_Segment_Any_3D_Objects_in_the_Wild_from_2D_ICCV_2025_paper.html) / [arXiv v2](https://arxiv.org/abs/2503.08407v2) | ICCV 2025; 94.1 NVOS, 94.0 SPIn | Reports all eight NVOS and all ten SPIn scenes, including Fork. Random prompt sampling and random selection of five viewpoints have no disclosed seed. SAM2 masks are precomputed from multiview RGB and cached. | Highest full-cohort broad online NVOS headline found, but not a frozen same-protocol comparator. SPIn prompt semantics remain underdocumented. |
| [SAGOnline](https://arxiv.org/abs/2508.08219v2) | arXiv v2, 2026-01-06; 92.7 NVOS, 95.2 SPIn | Uses 30k 3DGS, pseudo-temporal RGB, SAM2, and random clicks sampled from the ground-truth object; point count/seed and auditable ten-scene aggregate are absent. | Ineligible for official NVOS scribbles, deterministic SPIn sparse points, and full-reference-mask comparison. Ground-truth-derived query generation also violates the proposed destination unless explicitly authorized and frozen. |
| [MV-SAM](https://arxiv.org/abs/2601.17866v1) | arXiv v1, 2026-01-25; 92.1 NVOS, 92.9 SPIn | Samples 8 positive and 2 negative points, but does not disclose the seed; excludes NVOS orchids and SPIn pinecone. | Not full 8/10 cohort and not deterministic. Its table's recomputed reduced-cohort baselines cannot set full-cohort targets. |
| [GaussianTrimmer](https://arxiv.org/abs/2601.12683) | arXiv 2026; up to 92.5 NVOS | Post-processes prior methods using SAM2 on rendered virtual RGB views; its baseline table uses the older SAGA track. | Online rendered-RGB postprocessing; not strict-unseen and not a reusable-field comparator. |
| [NG-GS](https://openaccess.thecvf.com/content/CVPR2026/html/He_NG-GS_NeRF-guided_3D_Gaussian_Splatting_Segmentation_CVPR_2026_paper.html) | CVPR 2026; 92.6 NVOS IoU, 99.2 accuracy, 84.7 boundary IoU | Reports eight scenes but begins from multiview mask signals produced by SAM or a trained mask model; official scribble-to-mask generation and seed are not frozen. | Online/multiview context, not strict-unseen same protocol. |
| [Online Segment 3D Gaussians via Launching Virtual Drones](https://arxiv.org/abs/2607.01628) | arXiv v1, 2026-07-02; 92.7 NVOS, 92.5 SPIn | Invokes SAM2 on RGB rendered from prompt-directed virtual views; supports click, box, and text prompts. It does not bind the official NVOS scribble or a deterministic SPIn point sampler. | Newest material increment before cut-off, but online prompt-specific RGB processing makes it ineligible for the destination's frozen-field primary track. |

### What changed materially

**Fact.** Later primary sources establish higher broad headline numbers than the
older registry context: WildSeg3D reaches 94.1% on all eight NVOS tasks, and
SAGOnline reports 95.2% on SPIn. The July 2026 virtual-drone paper is the newest
material publication found before the cut-off.

**Inference.** None of these increments supplies a new exact comparator for
`nvos_strict_unseen_v1`, `spin_nerf_point_prompt_10scene_v1`, or LUDVIG's
full-reference-mask protocol. Their prompt generation, view visibility, query-time
models, or cohort identity differ. Consequently, replacing a frozen target with the
largest headline would create a protocol error rather than raise the bar fairly.

## Current RADIO-GS deltas

All differences below are percentage points of macro foreground IoU. A signed delta
is `RADIO-GS - comparator`. "Not computable" is intentional when the evaluated
cohort or prompt is not identical.

| Track | RADIO-GS / validated local row | Comparator | Delta | Interpretation |
| --- | ---: | ---: | ---: | --- |
| NVOS strict-unseen, full 8 | 74.777124 | Original NVOS 70.1 | **+4.677124** | Provisional lead, pending exact original split/manifest binding. |
| NVOS strict-unseen, internal promotion gate | 74.777124 | Repository gate 74.9 | **-0.122876** | Engineering gate only; it is not a published SOTA comparator. The 75.066273 source-RGB-sidecar result is an illegal upper-bound diagnostic for the destination. |
| NVOS online-multiview, exact LUDVIG-SAM full 8 | 91.257685 | LUDVIG-DINOv2 92.4 | **-1.142315** | Same broad LUDVIG paper protocol, different 2D feature carrier. Diagnostic only. |
| NVOS online-multiview, exact LUDVIG-SAM full 8 | 91.257685 | WildSeg3D 94.1 | **-2.842315** | Cross-protocol context only; random prompt/five-view seeds are missing. |
| SPIn full-mask, RADIO-GS local 9 | 87.716150 | Exact LUDVIG-SAM local 9, 93.720045 | **-6.003895** | Same available nine-scene cohort; RADIO-GS primary is RGB-free, whereas LUDVIG is query-RGB-assisted. |
| SPIn full-mask, full 10 incl. Fork | not available | LUDVIG paper 93.8 | **not computable** | Fork is missing locally; do not extrapolate it from the nine-scene macro. |
| SPIn sparse-point, full 10 | not available | no exact comparator | **not computable** | Point budget/sampler/seed are not frozen. |

The RADIO-GS values come from
`paper/artifacts/unified_six_task_single_radio_mainline_v2.yaml`. The NVOS primary
uses only the canonical RADIO field at query time. The SPIn local-nine primary is
also RGB-free; its target-RGB-SAM diagnostic is explicitly not the primary track.

## Dated SOTA Target candidates

These are the targets that may enter the wayfinder map as of **2026-08-11**. Their
status is part of the target and must not be dropped when copied into another issue.

| Track | Candidate target | Status and registration rule |
| --- | --- | --- |
| NVOS strict-unseen | **>= 70.1% macro foreground IoU, official 8-task cohort**, original NVOS CVPR 2022 | **Provisional same-protocol candidate.** Register as an external published floor only after exact image/camera/scribble/evaluation identity is bound. RADIO-GS's `>=74.9%` remains a separate internal promotion gate, not external SOTA. |
| NVOS online-multiview | **>= 92.4% macro IoU** for the LUDVIG all-view protocol (DINOv2); **91.3375%** if the comparator is restricted to LUDVIG-SAM's recomputed same-eight macro | **Eligible diagnostic target.** Must be labeled target/all-view-RGB and query-2D-model assisted. WildSeg3D 94.1% may be recorded only as a broad online headline until its random prompt/view seeds are recoverable. |
| SPIn sparse-point, 10 scenes incl. Fork | **No numeric target yet** | **Blocked.** First freeze point labels, positive/negative budget, replacement policy, seed list, reference-view identity, all ten scenes, metric, and aggregation. Then reproduce at least one comparator on that exact sampler. |
| SPIn full-reference-mask, 10 scenes incl. Fork | **>= 93.8% macro foreground IoU**, LUDVIG ICCV 2025 | **Provisional online/full-mask candidate.** Acquire or reproduce Fork and bind a full-ten result before computing the RADIO-GS delta. The local-nine 93.720045% remains the exact available-cohort comparator. |

This audit deliberately does not nominate 94.1% NVOS or 95.2% SPIn as primary
targets. They answer different questions. A future target registry may contain them
under separately named online/random-prompt tracks, but it must not use them as
evidence for a frozen compact field that cannot read RGB or invoke a foundation model
at query time.

## Required resolutions before `/to-spec`

1. **Bind original NVOS identity.** Hash or otherwise freeze the original release's
   eight task manifests, reference/target image names, scribbles, target masks,
   camera conventions, preprocessing, and aggregation against
   `nvos_strict_unseen_v1`. Until then, use “provisional lead,” not “strict-unseen
   SOTA.”
2. **Freeze SPIn sparse prompts as a new explicit protocol.** Specify the exact ten
   scenes including case-sensitive `Fork`, reference frame per scene, positive and
   negative point counts, sampler and boundary policy, coordinate convention,
   replacement behavior, public seed list, query-visible inputs, frame/scene
   aggregation, and number of runs. Do not infer SAGA's undisclosed sampler.
3. **Close the Fork gap.** Obtain the missing Fork assets and run the same frozen
   full-mask LUDVIG/RADIO-GS evaluation. Until then, preserve local-nine and
   published-full-ten as different rows.
4. **Repair the launcher provenance binding.** Either run exact reproductions from
   the historical launcher whose hash is in the registry, or add a reviewed binding
   that proves the current launcher's default path is protocol-equivalent while
   preserving the historical result hash.
5. **Name visibility tracks in every result.** At minimum, distinguish
   `strict_unseen_field_only`, `online_all_view_query_fm`,
   `sparse_point_frozen_sampler`, and `full_reference_mask`. Every target must record
   whether target RGB, rendered RGB, cached masks, and query-time external models are
   allowed.
6. **Do not tune on test labels.** Reference masks may be benchmark-authorized task
   inputs only in the full-mask track. Target masks remain scoring-only. Any paper
   that samples clicks from ground-truth objects or selects thresholds on reference
   GT must be represented exactly as such, never silently treated as a user prompt.

## Decision-ready conclusions

- **Fact:** the repository's immutable NVOS full-eight and SPIn local-nine LUDVIG
  summaries remain valid artifacts; the current launcher no longer has their frozen
  byte identity.
- **Decision candidate:** preserve strict-unseen and online-multiview as separate
  benchmark families permanently.
- **Decision candidate:** preserve SPIn sparse-point and full-reference-mask as
  separate benchmark families permanently.
- **Decision candidate:** a published number cannot be a SOTA Target unless prompt,
  cohort, visibility, query-time compute, metric, and aggregation identities are all
  frozen; otherwise it is contextual evidence only.
- **Unknown:** exact same-protocol public SOTA above the original NVOS 70.1% was not
  found for the strict-unseen track by the cut-off.
- **Unknown:** no published deterministic full-ten SPIn sparse-point target was found
  by the cut-off.
- **Unknown:** no locally reproduced Fork result exists for the full-mask ten-scene
  track.
