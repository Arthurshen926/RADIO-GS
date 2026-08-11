# ScanNet OVS and AGILE3D protocol/SOTA delta audit

- Research cutoff: **2026-08-11 (Asia/Shanghai)**
- Wayfinder ticket: [#2](https://github.com/Arthurshen926/RADIO-GS/issues/2)
- Scope: protocol identity, comparator provenance, incremental primary-source changes, and dated SOTA Target candidates only
- Evidence notation: **Fact** is directly supported by a cited artifact or primary source; **Inference** is a conclusion drawn from those facts; **Unknown** is not fixed by the available evidence.

## Decision summary

1. **Reuse the two existing Validated Protocol Artifacts.** Their immutable result hashes, complete cohorts, evaluator contracts, and source/checkpoint pins pass the current audit. Reuse does not erase their recorded comparability qualifications.
2. **Freeze ScanNet OVS to VALA paper8, not the repository's later code9 list.** The benchmark cohort is the eight named scenes and the headline is six numbers: mIoU/mAcc for the 19/15/10 class splits, scene-equal macro aggregation.
3. **Freeze AGILE3D to the official full312 cohort.** A result is headline-eligible only after all 312 validation scenes and all 10,357 released target objects are evaluated under one frozen click/evaluator contract. The current RADIO-GS 20-scene, 804-object dense-RGB-D overlap run is a development pilot and has no valid SOTA delta.
4. **Do not promote E2I3D into the frozen AGILE3D target yet.** It is a material 2026 paper, but it does not disclose the full312/10,357 identity fields or released preprocessing/evaluator hashes, reports only clicks 5/10/15, and does not beat its own AGILE3D row on ScanNetV2 accuracy.
5. **Recommended dated target semantics:** use a component-wise same-protocol envelope, with no unstated tolerance. For ScanNet that envelope is the stronger local exact VALA reproduction. For AGILE3D it is the maximum of the validated full312 Easy3D reproduction and the published Easy3D/AGILE3D table at each frozen click count.

## Validated Protocol Artifact checks

### ScanNet OVS / VALA

**Fact.** The canonical repository artifact is [`concept_scannet_ovs_vala_paper8`](../../../paper/artifacts/evaluation_protocol_freeze_20260801.yaml), backed by the exact-protocol report [`vala_scannet_exact_protocol_reproduction_20260801.md`](../../../paper/artifacts/vala_scannet_exact_protocol_reproduction_20260801.md) and authoritative result SHA-256 `81e6584a29eab59ffacba91b21d079c88213643a1b9a234356240d70b7d13740`. The hash was recomputed from the retained result and matches the freeze.

**Fact.** The frozen semantic/evaluator path uses the upstream VALA checkout named for commit `48902a541333d65aeb0aebf64ad664777a27c3fc`. That commit is still the official repository `main` head at the cutoff. Its July 23 change made the ScanNet runner executable and replaced a path-name heuristic with explicit ScanNet gates `tau_mass=0.9`, `tau_abs=0.01`; it did not publish a replacement metric table ([official commit](https://github.com/changandao/VALA/commit/67f202e89ae98c9e26f6ca41052fc53516ab18b1)). The frozen artifact uses those same gates.

**Fact.** The external paper identity is VALA, arXiv `2509.05515v2`, dated 2026-02-10 in the PDF and listed by the authors as 3DV 2026. Table 2 reports the same six paper values stored by the freeze: `(32.11, 50.05)`, `(35.10, 54.77)`, `(46.21, 65.61)` for 19/15/10-class mIoU/mAcc ([VALA v2](https://arxiv.org/abs/2509.05515v2), [official project](https://vala3d.github.io/)).

**Fact.** The reproduction is protocol-exact for semantic lifting and evaluation but is not a fresh end-to-end paper geometry reproduction: it uses available local RGB Gaussian geometry. The repository therefore marks a direct paper-table claim `diagnostic_only`, even though every recorded protocol-match field is true ([registry row](../../../paper/artifacts/evaluation_protocol_registry_20260731.yaml)).

**Inference.** The artifact is valid for fixing the protocol and for an internal same-evaluator comparator. A publication claim should separately disclose the compatible local geometry provenance.

**Unknown.** The ScanNet row in the compact freeze does not itself contain an explicit `source_commit` field (unlike the adjacent LERF-3D VALA row). The retained execution paths identify `48902a5`, but `/to-spec` should copy the full commit into the ScanNet contract rather than rely on a directory name.

### AGILE3D ScanNet40 / Easy3D

**Fact.** The canonical artifact is [`spatial_agile3d_easy3d`](../../../paper/artifacts/evaluation_protocol_freeze_20260801.yaml), backed by [`easy3d_agile3d_protocol_audit_20260731.md`](../../../paper/artifacts/easy3d_agile3d_protocol_audit_20260731.md). Its primary full-cohort result hash is `c771f29400e912565ee2ea5a754d0fd80a7fafc1eb91e38b6db1953cdcbbc09d`; the released-Easy3D-forward sensitivity hash is `c8c0c6820cf88166e77002a20ddf68af44c7be2e58a2e36cd26b147bdbd2f5a1`; and the preprocessing manifest hash is `39035ec87a3ff73bd9cfd6eec9a93182b7ebd7d9b2e84515b1c0e51cad453d23`. All three were recomputed and match the freeze.

**Fact.** The provenance pins Easy3D commit `b3f5bd70defaa9a601edb0975802775b056c784a`, AGILE3D commit `b73638da41edbabe52a1b578d52ddeb8fa552173`, and Easy3D checkpoint SHA-256 `4a13d16ba2f2470031287812dbbdf1ec6aa14097cb3738e0fe596bb708dc475f`. Both pinned commits remain the official repositories' heads at the cutoff ([Easy3D official repository](https://github.com/facebookresearch/easy3d/tree/b3f5bd70defaa9a601edb0975802775b056c784a), [AGILE3D official repository](https://github.com/ywyue/AGILE3D/tree/b73638da41edbabe52a1b578d52ddeb8fa552173)).

**Fact.** The full run covers 312 scenes and 10,357 released objects with zero failures. It uses released Easy3D stable last-write voxel quantization, integer 5-cm voxel coordinates, fixed sigmoid threshold `0.5`, the AGILE3D release click policy, point-level IoU after clicks 1/2/3/5/10, and query-micro averaging over all 10,357 objects.

**Fact.** The official Easy3D paper says it follows AGILE3D's single-object protocol, released preprocessed data, selected instances, official evaluator, and 5-cm voxels. Its ScanNet40 table reports Easy3D `68.2/74.6/77.3/79.6/81.7` and AGILE3D `63.0/70.6/75.1/79.7/83.5` at clicks 1/2/3/5/10 ([ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Simonelli_Easy3D_A_Simple_Yet_Effective_Method_for_3D_Interactive_Segmentation_ICCV_2025_paper.pdf)). The official AGILE3D repository states that it releases the processed benchmark data, result CSVs, and single-object evaluation entrypoint ([AGILE3D repository](https://github.com/ywyue/AGILE3D)).

**Fact.** The local audit preserves, rather than hides, four incompatibilities: paper prose says voxel RGB is averaged while released Easy3D code uses stable last-write quantization; Easy3D and AGILE3D corrective forwards differ; the release contains 10,357 objects while the bundled legacy result-key intersection contains 10,016; and threshold/checkpoint calibration is under-specified by the paper table. Accordingly the registry marks the reproduction `diagnostic_only` for a literal paper-table claim.

**Inference.** This is still the strongest auditable comparator for RADIO-GS because cohort membership, preprocessing, clicks, threshold, aggregation, and immutable outputs are explicit. “Diagnostic-only versus the paper” does not mean “invalid for internal target setting.”

## Frozen protocol identities

### ScanNet OVS paper8

**Fact.** The cohort is exactly:

`scene0000_00`, `scene0062_00`, `scene0070_00`, `scene0097_00`, `scene0140_00`, `scene0347_00`, `scene0400_00`, `scene0590_00`.

`scene0645_00` is a later code9 sensitivity scene and is forbidden from the headline cohort.

**Fact.** The remaining identity fields are:

- splits: fixed 19/15/10 ScanNet class-name queries encoded with the released CLIP text cache;
- prediction domain: optimized/pruned Gaussian rows at full resolution, feature level 0;
- lift: official gsplat alpha-times-transmittance significance, stochastic robust aggregation, `tau_mass=0.9`, `tau_abs=0.01`, significance cutoff `1e-5`;
- pseudo-GT: anisotropic Mahalanobis Gaussian assignment, search radius five times maximum Gaussian scale, top-k 1000, class-balanced vote;
- metric weight: `sigmoid(opacity) * scale_x * scale_y * scale_z`;
- class aggregation: only classes present in ground truth for that scene;
- final aggregation: unweighted equal macro over the same eight scenes;
- reported tuple order: `19-mIoU, 19-mAcc, 15-mIoU, 15-mAcc, 10-mIoU, 10-mAcc`.

Any change to one of these fields is a new protocol ID, not a new result under `concept_scannet_ovs_vala_paper8`.

### AGILE3D official full312

**Fact.** The frozen full benchmark identity is:

- ScanNet40 validation release: 312 scenes, 10,357 selected object queries, zero skipped/failed objects;
- 5-cm released voxel representation, stable last-write quantization, Easy3D integer-voxel coordinates;
- first click at the center of the largest false-negative region;
- each correction at the center of the largest false-positive or false-negative region, choosing false-positive on an exact radius tie;
- clicked voxel overwritten with its ground-truth label by the evaluator;
- fixed sigmoid threshold `0.5`, maximum 10 clicks, BF16, object batch size 4;
- point-level object IoU at clicks `1, 2, 3, 5, 10`;
- query-micro mean over all 10,357 object trajectories.

**Fact.** Ground truth is authorized only inside the benchmark harness to generate simulated corrective clicks and compute metrics. It is not a scene-feature or model input. This is ordinary interactive-benchmark evidence, not permission for target-mask-conditioned calibration.

## Full312 versus the 20-scene pilot

| Field | Official headline | Current RADIO-GS pilot |
|---|---:|---:|
| Scenes | 312 | 20 fixed overlap scenes |
| Objects | 10,357 | 804 |
| Observation source | released ScanNet40 point release | dense registered RGB-D fields (117–240 views per scene are available in the pilot dataset) |
| Max clicks | 10 | 20 |
| Required headline clicks | 1/2/3/5/10 | includes 1/2/3/5/10/15 and NoC |
| Status | SOTA-eligible when complete | development-only, never a full312 claim |

**Fact.** The distinction is already specified in [`agile3d_scannet40_point_prompt.md`](../../benchmarks/agile3d_scannet40_point_prompt.md). The current pilot result at `output/agile3d_scannet40/canonical_dense20_cellseed_v1/results.json` reports `37.2015/32.6467/34.4196/42.5655/55.8011` at clicks 1/2/3/5/10 over 20 scenes and 804 objects.

**Inference.** Subtracting the full312 Easy3D row produces apparent gaps of `-32.5121/-43.3668/-44.1746/-38.4296/-27.2373` percentage points, but those are **non-comparable diagnostics** because both the cohort and observation contract differ. The correct official delta is **unknown until RADIO-GS completes full312**.

## Incremental primary-source audit through 2026-08-11

### ScanNet OVS

**Fact.** VALA v2 and the official 3DV 2026 project remain the latest primary source located for this exact Gaussian-domain, paper8, 19/15/10 ScanNet semantic protocol. The v2 paper's strongest row remains VALA itself. Searches across the official arXiv/OpenReview paper record and official code found no later primary source reporting a stronger result with the same eight-scene cohort, Gaussian pseudo-GT, weights, and scene-equal aggregation.

**Fact.** The only material post-paper repository change found was the July 23 official-code reproducibility repair described above. It preserves the paper ScanNet gate values and does not publish a new result tuple.

**Unknown.** Absence of a located result is not proof that no unpublished or differently indexed method exists. The target is therefore dated, not timeless.

### AGILE3D

**Fact.** A new primary source exists after Easy3D: E2I3D, AAAI 2026. It says other settings follow AGILE3D, trains on ScanNetV2-Train, evaluates on ScanNetV2-Val, and reports single-object mean IoU with at most 20 clicks. At clicks 5/10/15 it reports AGILE3D `79.9/83.7/85.0`; its largest 40.5M E2I3D model reports `79.9/83.6/85.0` ([AAAI 2026 paper](https://ojs.aaai.org/index.php/AAAI/article/download/37339/41301), [official proceedings record](https://ojs.aaai.org/index.php/AAAI/article/view/37339)).

**Fact.** E2I3D does not report clicks 1/2/3, cohort cardinalities, object-key identity, released preprocessing hash, prediction threshold, or an official code/checkpoint link in the primary paper record located by this audit. Its AGILE3D values also differ slightly from Easy3D's published AGILE3D row (`79.7/83.5` at clicks 5/10).

**Inference.** E2I3D is relevant evidence and a future protocol-reproduction candidate, but not a validated same-protocol replacement. Its own ScanNetV2 table does not establish a higher-accuracy method than AGILE3D at 5/10/15 clicks; its contribution is primarily efficiency.

## Strongest comparable rows

### ScanNet OVS

| Comparator | 19 mIoU / mAcc | 15 mIoU / mAcc | 10 mIoU / mAcc | Role |
|---|---:|---:|---:|---|
| VALA v2 paper | 32.11 / 50.05 | 35.10 / 54.77 | 46.21 / 65.61 | external published SOTA row |
| Validated local VALA exact protocol | 34.526858 / 51.590646 | 37.960598 / 56.769618 | 47.364163 / 67.465001 | stronger same-evaluator guardrail |
| RADIO-GS frozen current | 37.863296 / 55.218917 | 41.979415 / 60.441404 | 52.340812 / 68.602587 | current project row |

**Fact.** Relative to the stronger local guardrail, the current RADIO-GS row is ahead by mIoU `+3.336438/+4.018816/+4.976648` points and mAcc `+3.628271/+3.671786/+1.137586` points for 19/15/10.

**Inference.** ScanNet OVS is not presently an accuracy frontier; its remaining blockers are claim/provenance eligibility and preserving this result under the final unified frozen-field contract.

### AGILE3D full312

| Comparator | IoU@1 | IoU@2 | IoU@3 | IoU@5 | IoU@10 | Role |
|---|---:|---:|---:|---:|---:|---|
| Easy3D paper | 68.2 | 74.6 | 77.3 | 79.6 | 81.7 | published single-method row |
| AGILE3D as reproduced in Easy3D paper | 63.0 | 70.6 | 75.1 | 79.7 | 83.5 | published late-click leader |
| Validated local Easy3D, AGILE release semantics | 69.713667 | 76.013522 | 78.594169 | 80.995103 | 83.038390 | same-evaluator full312 guardrail |
| E2I3D paper | not reported | not reported | not reported | 79.9 | 83.6 | not protocol-validated; not stronger than its AGILE3D row |

The validated full312 Easy3D reproduction is strongest at clicks 1/2/3/5. The published AGILE3D row is strongest at click 10. A single-method comparison should name Easy3D and keep its full vector; a claim of “SOTA-level at every frozen click” should use the component-wise envelope below.

## Dated SOTA Target candidates

These candidates intentionally specify exact numeric boundaries. If a statistical or numerical tolerance is desired, it must be preregistered as an additional contract field; this audit does not invent one after seeing results.

### Candidate `SOTA-SCANNET-OVS-20260811`

- Protocol ID: `concept_scannet_ovs_vala_paper8` exactly as defined above.
- External comparator identity: VALA, arXiv `2509.05515v2` / 3DV 2026.
- Acceptance tuple in percent, ordered `19-mIoU, 19-mAcc, 15-mIoU, 15-mAcc, 10-mIoU, 10-mAcc`:
  **`[34.526858, 51.590646, 37.960598, 56.769618, 47.364163, 67.465001]`**.
- Rule: meet or exceed every component in one complete frozen paper8 run. This conservative tuple is the component-wise maximum of the published VALA row and the validated same-protocol VALA reproduction.
- Current status: RADIO-GS passes numerically, subject to the final unified-field eligibility/provenance audit.

### Candidate `SOTA-AGILE3D-FULL312-20260811`

- Protocol ID: `spatial_agile3d_easy3d` with `interaction_contract=agile3d_release` exactly as defined above.
- Comparator identities: Easy3D official ICCV 2025 checkpoint reproduction plus the published AGILE3D late-click row in the Easy3D paper.
- Acceptance tuple in percent, ordered `IoU@1, IoU@2, IoU@3, IoU@5, IoU@10`:
  **`[69.713667, 76.013522, 78.594169, 80.995103, 83.500000]`**.
- Rule: meet or exceed every component in one complete 312-scene, 10,357-object run with zero omissions. The first four components come from the stronger validated full312 Easy3D row; IoU@10 comes from the published AGILE3D row.
- Current status: **unknown/not run** for RADIO-GS. The 20-scene pilot cannot pass or fail this target.

### Optional cleaner single-method AGILE3D claim

If `/to-spec` chooses method-level comparability over a component-wise SOTA envelope, use only the validated Easy3D tuple:

**`[69.713667, 76.013522, 78.594169, 80.995103, 83.038390]`**.

This is easier to explain and reproduce, but it does not equal the best published IoU@10 value. It should be called “Easy3D-level” rather than “SOTA at every click.”

## Decisions and follow-up gates for the wayfinder map

1. **Accept:** paper8 and full312 are the only headline cohorts. Code9 and dense20 are sensitivity/development cohorts.
2. **Accept:** reuse both validated artifacts and their immutable hashes; retain every recorded comparability caveat.
3. **Recommended decision:** choose the component-wise target candidates above when the destination means SOTA-level at every reported metric. Choose the single-method alternative only if the intended claim is “matches or exceeds a named method.”
4. **Before `/to-spec`:** copy VALA commit `48902a541333d65aeb0aebf64ad664777a27c3fc` explicitly into the ScanNet protocol row.
5. **Before an AGILE3D claim:** execute RADIO-GS on all 312 scenes/10,357 objects under the frozen full contract. No conclusion can be extrapolated from dense20.
6. **Watch/research gate:** E2I3D may enter a future target only after a source/checkpoint release or independent audit proves cohort, preprocessing, interaction, threshold, and aggregation identity. Its current paper is insufficient to mutate the frozen target.

## Primary sources

- Wang et al., [Visibility-Aware Language Aggregation for Open-Vocabulary Segmentation in 3D Gaussian Splatting, arXiv v2](https://arxiv.org/abs/2509.05515v2), and [official VALA project](https://vala3d.github.io/).
- VALA official repository, [July 23, 2026 ScanNet reproducibility repair](https://github.com/changandao/VALA/commit/67f202e89ae98c9e26f6ca41052fc53516ab18b1).
- Simonelli et al., [Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation](https://openaccess.thecvf.com/content/ICCV2025/papers/Simonelli_Easy3D_A_Simple_Yet_Effective_Method_for_3D_Interactive_Segmentation_ICCV_2025_paper.pdf), ICCV 2025.
- [Easy3D official code at the audited commit](https://github.com/facebookresearch/easy3d/tree/b3f5bd70defaa9a601edb0975802775b056c784a).
- Yue et al., [AGILE3D official project](https://ywyue.github.io/AGILE3D/) and [official code at the audited commit](https://github.com/ywyue/AGILE3D/tree/b73638da41edbabe52a1b578d52ddeb8fa552173), ICLR 2024.
- Cong et al., [Towards Efficient and Effective Interactive 3D Segmentation](https://ojs.aaai.org/index.php/AAAI/article/download/37339/41301), AAAI 2026, with [official proceedings metadata](https://ojs.aaai.org/index.php/AAAI/article/view/37339).
