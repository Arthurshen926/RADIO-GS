# Five-benchmark core mainline — 2026-08-15

This is the single active experiment record for the post-grill cleanup. It
records development evidence, not a final five-benchmark claim.

The frozen joint-development identity is
`paper/artifacts/five_benchmark_method_v1_authority_20260815.json`. The older
five-contract gap matrix remains useful as an evaluation inventory, but its
historical compiler identity no longer defines the candidate method.

The live materialization gap is recorded in
`paper/artifacts/five_benchmark_method_v1_asset_inventory_20260815.json` and
enforced per field by
`python -m radio_gs.scripts.validate_five_benchmark_method_v1_field`.

## Frozen method decisions

- The persistent scene state is the schema-v2 factorized D512/L512 canonical
  RADIO field. Target reliability is an objective weight, not deployment state.
- Raw RADIO reconstruction owns the radial/log-amplitude gauge. SigLIP2,
  DINO, SAM, and region-summary objectives receive tangent-only gradients.
- Official SigLIP2 spatial projection runs on the complete 2-D token grid.
  Applying the SigLIP2 summary head independently to pixels or Gaussians is a
  retired proxy and is legacy-opt-in only.
- A valid region branch predicts a RADIO summary token from a region, then
  applies the frozen official summary head. Its teacher is generated from
  genuine source-RGB crops. Benchmark RGB, masks, queries, and labels are not
  field-training inputs.
- LERF primitive and region scores may be combined only by the global frozen
  1:1 rule in this development slice; no scene or query weight selection is
  permitted.

## Figurines source-only development slice

All runs use the four frozen LERF-OVS benchmark frames 41, 105, 152, and 195
only after training and candidate selection. The 295-frame feature and crop
teacher bundles exclude those frames.

| Candidate/readout | Raw validation | SigLIP2 spatial | Region summary | MPR probe | LERF2D mIoU | LocAcc |
|---|---:|---:|---:|---:|---:|---:|
| Initial D512/L512 field, primitive | 0.57957 | 0.62781 | — | 0.99362 | 0.37238 | 0.8750 |
| + 64-step official spatial, primitive | 0.58120 | 0.63653 | — | 0.99364 | 0.37298 | 0.8750 |
| + 64-step genuine region teacher, primitive | 0.58260 | 0.64182 | 0.50021→0.50914 | 0.99361 | 0.37394 | 0.8571 |
| Same field, region only | — | — | — | — | 0.17289 | 0.5357 |
| Same field, fixed 1:1 primitive+region scores | — | — | — | — | 0.34884 | 0.8750 |
| + generic response, frozen basis, primitive | 0.58363 | 0.64510 | 0.51885 | 0.99356 | 0.37558 | 0.8571 |
| + generic response, joint basis, early-stop step 1, primitive | 0.58564 | 0.64703 | 0.52567 | 0.99347 | 0.37580 | 0.8571 |

The generic response objective uses 806 target-blind generic text directions,
657 synonym directions, and 167 sibling relations. With the factorized basis
frozen, its validation loss falls from 0.32233 to 0.29122, response-profile
cosine rises from 0.11134 to 0.19667, and region validation rises from 0.50914
to 0.51885. Frozen LERF2D mIoU rises by only 0.00163 over the preceding field.
The same checkpoint's region-only and fixed 1:1 outputs fall to 0.15772 and
0.34517 mIoU respectively, so the current region readout remains rejected.

Joint basis optimization produces a slightly stronger early-stop checkpoint,
but its LERF2D increment over frozen-basis response training is only 0.00022.
Continuing to step 64 drops raw validation from 0.58260 to 0.57177 and MPR from
0.99361 to 0.98552. The persistent basis therefore remains frozen by default;
the one-step result is retained as an ablation, not promoted as a new method.

## Readout conclusion

The field objectives now preserve official spatial, genuine crop-summary, and
generic text-response capabilities without benchmark vocabulary. Their
source-only validation gains do not translate into a material LERF2D gain.
The demonstrated bottleneck is the typed region/query readout, not missing
field capacity. Per-scene fusion tuning, graph repair, and connected-component
repair remain prohibited because they would conceal rather than solve that
interface error.

## Frozen Method-v1 LERF full-four results

All four shared LERF fields now pass the executable Method-v1 gate, including
the exact predecessor-file hashes for base, official spatial, and genuine
crop-summary stages. The first completed full-four evaluation rendered the
RADIO field and then applied the SigLIP2 head. It is retained as a useful
dense-readout diagnostic, but it is not the frozen Method-v1 readout and is
not eligible for a Method-v1 row.

| Scene | Samples | LocAcc | sample mIoU |
|---|---:|---:|---:|
| Figurines | 56 | 0.85714 | 0.37559 |
| Ramen | 71 | 0.81690 | 0.20185 |
| Teatime | 59 | 0.89831 | 0.37762 |
| Waldo Kitchen | 22 | 0.81818 | 0.29908 |
| Full four diagnostic | 208 | 0.85096 | 0.30877 |

The diagnostic scene-macro mIoU is 0.31353 and category-macro mIoU is 0.30849.

The exact frozen Method-v1 LERF2D readout scores text on each primitive and
then renders the scalar score, with no primitive-confidence modification and
no mask refinement:

| Scene | Samples | LocAcc | sample mIoU |
|---|---:|---:|---:|
| Figurines | 56 | 0.91071 | 0.37021 |
| Ramen | 71 | 0.85915 | 0.21985 |
| Teatime | 59 | 0.89831 | 0.37536 |
| Waldo Kitchen | 22 | 0.81818 | 0.31187 |
| Full four Method-v1 | 208 | 0.87981 | 0.31417 |

The Method-v1 LERF2D scene-macro mIoU is 0.31932 and category-macro mIoU is
0.31303. This exact readout is slightly stronger than the dense diagnostic,
but it is not a SOTA result; Ramen remains the largest 2-D bottleneck.

The first `vala_paper_3d` evaluation below uses one primitive semantic level,
the fixed 0.6 score threshold, selected-only alpha projection, the released
`>10/255` silhouette rule, and no postprocessing. It is a Primitive
Readout-v0 diagnostic, not the completed frozen LERF3D protocol: the frozen
contract requires three semantic levels, kNN=10, per-query min-max mapping,
and highest-peak level selection.

| Scene | Samples | mIoU | Acc@0.25 | Acc@0.50 |
|---|---:|---:|---:|---:|
| Figurines | 56 | 0.44840 | 0.78571 | 0.44643 |
| Ramen | 71 | 0.29173 | 0.50704 | 0.21127 |
| Teatime | 59 | 0.32871 | 0.54237 | 0.22034 |
| Waldo Kitchen | 22 | 0.19815 | 0.36364 | 0.18182 |
| Full four Primitive Readout-v0 | 208 | 0.33450 | 0.57692 | 0.27404 |

The one-level scene-macro mIoU is 0.31675. Restoring only the frozen
query-independent kNN10 plus per-query scene min-max extent operator on the
same current field raises full-four sample-micro mIoU from 0.33450 to 0.46184
and scene-macro mIoU from 0.31675 to 0.44312. All four scenes improve. This
isolates protocol/readout incompleteness—not D512/L512 field damage—as the
main cause of the unexpectedly low row. The one-level corrected result remains
mechanism evidence only; the exact current-field three-level evaluation is the
next hard gate before any LERF3D final claim.

Waldo also exposed a legitimate grid-rounding boundary: the frozen crop
teacher is 46x62 while its native RADIO grid is 45x62. The shared semantic
alignment interface now permits only a one-cell full-extent bilinear alignment
and continues to reject larger or channel mismatches. Both region fidelity and
generic response use the same helper. The original Waldo construction then
completed, and the final generic stage improved generic loss 0.29299 to
0.27985, profile cosine 0.18656 to 0.22345, and region validation 0.55012 to
0.55796 while retaining MPR within 0.00003.

## ScanNet OVS Method-v1 paper-eight cohort

Scene0070 was the first paper-eight ScanNet scene rebuilt as the exact
Method-v1 field. Scene0000, scene0062, scene0097, scene0140, scene0347,
scene0400, and scene0590 reproduce the same source-only chain. In scene order
0000, 0062, 0070, 0097, 0140, 0347, 0400, and 0590, their deterministic
fidelity holdouts are respectively frames 1100/2220/3340/4460,
140/280/440/580, 260/520/800/1060, 140/300/440/600,
860/1720/2580/3440, 200/420/640/860, 280/520/760/1000, and
540/1080/1620/2160. The exact-marginal base construction excludes only those
four views in each scene. All eight final fields pass the strict schema-v2
D512/L512 lineage gate, and every query-independent primitive cache is
SHA-bound to its field and geometry.

For the original five-scene slice, the three capability stages all improve
their own held-out objective while retaining the exact-marginal MPR probe:

| Stage | Primary held-out change | Raw validation | MPR probe |
|---|---:|---:|---:|
| Official SigLIP2 spatial | 0.74106 → 0.74520 | 0.75649 → 0.75703 | 0.99566 → 0.99567 |
| Genuine crop summary | 0.53660 → 0.54418 | 0.75703 → 0.75739 | 0.99567 → 0.99565 |
| Generic text response | 0.29498 → 0.27791 loss; 0.18467 → 0.23178 cosine | 0.75739 → 0.75764 | 0.99565 → 0.99561 |

The frozen Gaussian-center evaluator consumes the external primitive cache
directly, checks its cache hash and embedded field provenance, row count, XYZ
alignment, feature dimension, validity mask, method identity, and
no-postprocessing contract, then applies
the fixed five-prompt SigLIP2 text ensemble. It performs no target-scene
calibration or prediction postprocessing:

| Split | VALA volume mIoU | VALA volume mAcc | Row mIoU | Row mAcc |
|---|---:|---:|---:|---:|
| 19 classes | 0.28825 | 0.49546 | 0.31334 | 0.49073 |
| 15 classes | 0.30541 | 0.44857 | 0.32720 | 0.43890 |
| 10 classes | 0.32357 | 0.69809 | 0.40309 | 0.70776 |

The scene0070 row above was the initial valid single-scene development
sentinel, not a paper-eight aggregate or a SOTA claim. At that checkpoint the
legal inventory was 5/29 overall and 1/8 for ScanNet.

Scene0097 and scene0347 then completed the identical capability chain:

| Scene/stage | Primary held-out change | Raw validation | MPR probe |
|---|---:|---:|---:|
| 0097 official SigLIP2 | 0.77265 → 0.77807 | 0.77587 → 0.77612 | 0.99744 → 0.99749 |
| 0097 genuine crop summary | 0.54404 → 0.55346 | 0.77612 → 0.77627 | 0.99749 → 0.99748 |
| 0097 generic response | 0.28744 → 0.26044 loss; 0.20324 → 0.27869 cosine | 0.77627 → 0.77629 | 0.99748 → 0.99741 |
| 0347 official SigLIP2 | 0.75276 → 0.75715 | 0.76032 → 0.76107 | 0.99663 → 0.99671 |
| 0347 genuine crop summary | 0.54795 → 0.55849 | 0.76107 → 0.76161 | 0.99671 → 0.99672 |
| 0347 generic response | 0.32283 → 0.29430 loss; 0.10467 → 0.18411 cosine | 0.76161 → 0.76200 | 0.99672 → 0.99667 |

Scene0400 and scene0590 complete two further instances with the same frozen
weights, step budget, and selection policies:

| Scene/stage | Primary held-out change | Raw validation | MPR probe |
|---|---:|---:|---:|
| 0400 official SigLIP2 | 0.75151 → 0.75500 | 0.74618 → 0.74703 | 0.99706 → 0.99708 |
| 0400 genuine crop summary | 0.55965 → 0.56811 | 0.74703 → 0.74771 | 0.99708 → 0.99707 |
| 0400 generic response | 0.29610 → 0.27534 loss; 0.17828 → 0.23606 cosine | 0.74771 → 0.74818 | 0.99707 → 0.99704 |
| 0590 official SigLIP2 | 0.76952 → 0.77176 | 0.77409 → 0.77463 | 0.99622 → 0.99626 |
| 0590 genuine crop summary | 0.57323 → 0.57831 | 0.77463 → 0.77503 | 0.99626 → 0.99626 |
| 0590 generic response | 0.28262 → 0.26834 loss; 0.21525 → 0.25491 cosine | 0.77503 → 0.77532 | 0.99626 → 0.99622 |

Scene0000 then completes the same chain without a CLI frame allowlist. The
configured authority selects exactly 275 train frames from the 279-frame
source bundle after excluding the four frozen fidelity frames:

| Scene/stage | Primary held-out change | Raw validation | MPR probe |
|---|---:|---:|---:|
| 0000 official SigLIP2 | 0.70774 → 0.71080 | 0.69755 → 0.69799 | 0.99464 → 0.99478 |
| 0000 genuine crop summary | 0.57030 → 0.57574 | 0.69799 → 0.69828 | 0.99478 → 0.99483 |
| 0000 generic response | 0.29160 → 0.28569 loss; 0.19509 → 0.21134 cosine | 0.69828 → 0.69836 | 0.99483 → 0.99481 |

Scene0062 was rebuilt from a current 37-frame source bundle and its v14
geometry. Its exact-marginal authority covers 33 training views after holding
out frames 140/280/440/580; 49,720 of 51,610 Gaussians receive valid source
observations. The three frozen stages improve every primary objective:

| Scene/stage | Primary held-out change | Raw validation | MPR probe |
|---|---:|---:|---:|
| 0062 official SigLIP2 | 0.73346 → 0.73972 | 0.75124 → 0.75171 | 0.99766 → 0.99774 |
| 0062 genuine crop summary | 0.54672 → 0.55769 | 0.75171 → 0.75205 | 0.99774 → 0.99774 |
| 0062 generic response | 0.34217 → 0.30761 loss; 0.05218 → 0.14803 cosine | 0.75205 → 0.75222 | 0.99774 → 0.99766 |

Scene0140 closes the paper-eight cohort from 215 registered frames and its v14
geometry. The exact-marginal authority uses 211 training views after excluding
frames 860/1720/2580/3440; 340,418 of 372,941 Gaussians have valid source
observations. Its three frozen stages also improve every primary objective:

| Scene/stage | Primary held-out change | Raw validation | MPR probe |
|---|---:|---:|---:|
| 0140 official SigLIP2 | 0.72301 → 0.72591 | 0.72078 → 0.72163 | 0.99680 → 0.99680 |
| 0140 genuine crop summary | 0.48987 → 0.49334 | 0.72163 → 0.72238 | 0.99680 → 0.99678 |
| 0140 generic response | 0.37603 → 0.36371 loss; -0.04068 → -0.00715 cosine | 0.72238 → 0.72300 | 0.99678 → 0.99675 |

Scene0400 also exposed a real cohort-authority defect: its source image
directory contains 64 extracted frames, but only 61 have registered training
poses. The finetune entrypoint previously ignored the configured
`train_frame_ids_path` unless `--include-frame-ids` was repeated on the command
line. It now resolves an explicit CLI allowlist first and otherwise requires
the configured frozen allowlist, failing closed on a missing or empty file.
The corrected path ran scene0590 without a CLI allowlist and selected exactly
its 135 registered frames.

Their frozen primitive-readout ScanNet metrics are:

| Scene | Split | VALA volume mIoU | VALA volume mAcc |
|---|---:|---:|---:|
| 0000 | 19 | 0.34441 | 0.70967 |
| 0000 | 15 | 0.32313 | 0.72529 |
| 0000 | 10 | 0.35116 | 0.79991 |
| 0062 | 19 | 0.27887 | 0.68145 |
| 0062 | 15 | 0.27821 | 0.68252 |
| 0062 | 10 | 0.36375 | 0.67099 |
| 0097 | 19 | 0.31038 | 0.72751 |
| 0097 | 15 | 0.29355 | 0.70608 |
| 0097 | 10 | 0.46517 | 0.69223 |
| 0140 | 19 | 0.31745 | 0.63286 |
| 0140 | 15 | 0.36366 | 0.60116 |
| 0140 | 10 | 0.41866 | 0.72316 |
| 0347 | 19 | 0.36930 | 0.71201 |
| 0347 | 15 | 0.35192 | 0.68383 |
| 0347 | 10 | 0.64514 | 0.78158 |
| 0400 | 19 | 0.42121 | 0.68529 |
| 0400 | 15 | 0.39972 | 0.69110 |
| 0400 | 10 | 0.37690 | 0.80205 |
| 0590 | 19 | 0.35296 | 0.67779 |
| 0590 | 15 | 0.35396 | 0.65779 |
| 0590 | 10 | 0.43270 | 0.74171 |
| Paper-eight macro | 19 | 0.33535 | 0.66525 |
| Paper-eight macro | 15 | 0.33370 | 0.64954 |
| Paper-eight macro | 10 | 0.42213 | 0.73871 |

The complete paper-eight macro is now eligible under the frozen reproduced
protocol. The legal inventory is 12/29 overall and 8/8 for ScanNet. This closes
the missing-scene/protocol gap; it does not establish SOTA, and the measured
row must now be compared against the exact frozen-paper targets.

## NVOS/SPIn query-transient RGB/SAM adapter

The common persistent/transient seam is now explicit in
`radio_gs/querying/transient_rgb_sam.py` and is emitted by both the audited
LUDVIG reproduction wrapper and the SPIn reference selector:

- the persistent field ends at a signed prompt and never stores target RGB or
  SAM state;
- ten deterministic trials use three positive and three negative points;
- target RGB and frozen SAM are query-transient, while target masks and target
  metrics are unavailable during proposal generation;
- SPIn may calibrate candidate/threshold on its one permitted reference mask;
  NVOS scribbles do not gain that full-mask calibration authority;
- exact positive/negative observations are clamped after SAM fusion; conflicts
  preserve the base posterior; no graph or connected component is applied.

Re-aggregating all nine sealed historical SPIn reports under the stricter
contract reproduces canonical 0.877162, SAM-only 0.946957, and reference-only
selected 0.948415 macro foreground IoU. The latter is +0.011214 over the local
LUDVIG-SAM reproduction. This validates the adapter and selector semantics,
but it is still a historical carrier rather than the new D512/L512 field.
NVOS's released-compatible target-RGB path remains the audited 0.912577
reference.

The complete D512/L512 NVOS full-eight cohort has now been regenerated. The
first frozen Method-v1 majority-vote readout reaches 0.526874 macro foreground
IoU and 0.929458 pixel accuracy. A causal audit shows that the rendered signed
field is substantially stronger than this row: sampled positive points are at
least 96.7% target-correct and all sampled negative points are correct, while
the majority-vote SAM output is consistently high-precision but low-recall.
This rules out D512/L512 field corruption as the primary cause of the low row.

A preregistered box-SAM3 candidate selected by whole coarse-mask overlap
reaches 0.731381 macro IoU but fails on `horns_left` by selecting the larger
center object. It is rejected. Its preregistered successor selects proposals
by inclusion/exclusion of all sealed signed points, using coarse overlap and
SAM confidence only as tie-breaks. It reaches 0.817617 macro IoU and 0.970204
pixel accuracy; `horns_left` rises from 0 to 0.689883 and the other seven
scenes are bit-for-bit unchanged. The candidate passes every preregistered
development gate and replaces majority vote as the current target-RGB-assisted
development readout. It is not strict-unseen eligible and is not a SOTA claim,
because transient SAM opens the target RGB.

## Five-benchmark readiness

- LERF2D: the legal D512/L512 full-four cohort is complete; the current
  0.31417 sample-micro mIoU demonstrates a readout-quality gap, not a missing
  field or protocol gap.
- LERF3D: the shared full-four field cohort and frozen typed 3-D evaluation are
  complete; 0.33450 sample-micro mIoU demonstrates the same readout-quality
  gap, with Waldo Kitchen as the worst scene.
- ScanNet OVS: all eight required scenes have complete D512/L512 Method-v1
  fields, primitive caches, and frozen results. The paper-eight macro is
  0.33535/0.33370/0.42213 mIoU for the 19/15/10-class splits. The row is
  protocol-eligible but does not yet support a SOTA claim.
- NVOS: all eight D512/L512 fields and the hash-bound full-eight evaluation are
  complete. Frozen majority vote is 0.52687 macro IoU; the promoted
  signed-evidence SAM selector is 0.81762 development macro IoU. The remaining
  hard gap is a target-RGB-free readout or an explicitly comparable protocol,
  not field materialization.
- Available-Nine SPIn-NeRF: frozen protocol exists; unified D512/L512 fields
  must replace the historical carrier. Legal reference-only transient
  selection and the full-nine barrier runner are implemented; D512/L512 field
  construction has started with orchids and leaves.

Real orchids timing exposed a storage-partition inefficiency in the 4096-D
exact-DINO cache: every 256-channel outer shard repeats the frozen official
adaptor projection for every registered view. Future newly started scenes use
512-channel outer shards while retaining the 128-channel accumulation chunk,
normalization, float32 accumulation, and float16 serialization. A synthetic
2-versus-4-channel fixture proves bitwise-equal reconstructed features,
support counts, and reliability. Already-running orchids/leaves keep their
original 256-channel resume contracts and are not restarted.

The first real orchids base-field preflight also exposed a fail-closed receipt
schema omission: the capability cohort recorded both
`reference_masks_opened=false` and `evaluation_masks_opened=false`, but lacked
the trainer's aggregate `benchmark_masks_opened=false` field. No tensor
training started and no partial field was written. The original receipt is
retained, while a versioned `method_v1_capability_cohort_authority_v2.json`
adds only the missing aggregate flag. The factorized/RADIO/DINO/SAM caches,
registration, field hyperparameters, and information-access boundary are
unchanged. Orchids subsequently passed the v2 preflight and completed its
D512/L512 base field; the post-field teacher/readout/gate stages remain active.

The first orchids SigLIP validation then exposed a separate resolution-scaling
defect: the complete 189x252 SigLIP2 attention grid was projected in float32,
requesting 135.21 GiB. The already sealed official extractor projects the same
complete grid under CUDA AMP. AMP fixes inference, but Torch 2.0.1 on sm86
cannot train its head_dim=80 flash/memory-efficient SDPA and otherwise creates
a 67.61 GiB half-precision attention matrix. The SPIn runner therefore opts
into both the extractor-matched AMP precision and xFormers 0.0.20 exact global
memory-efficient attention. No tiling, truncation, or attention approximation
is used, and existing callers retain their prior default. Real complete-grid
inference and gradient smoke tests peak at 2.21 GB and 5.84 GB respectively;
the gradient is finite and nonzero.

The following orchids region stage exposed a second, independent allocator
boundary. AdamW moments remained on GPU while dense semantic/capability graphs
were built, first failing at step 2 and then—after validation-cache cleanup—on
the forward immediately after step 32. Dense step references are now released
before the next forward, unused CUDA blocks are returned after validation, and
AdamW moments live on CPU between steps while using the unchanged GPU update at
`optimizer.step()`. The corrected real run completed all 64 steps. Source-only
semantic validation rose from 0.422946 to 0.439117, raw/SigLIP validation also
rose, and the MPR probe changed from 0.999107 to 0.999099, well inside the
frozen 0.0002 drop gate. The resulting field has entered its final target-blind
generic-response stage; no benchmark masks or scores were opened.

Leaves also exposed two execution-boundary defects without invalidating its
completed base field. Crop-summary extraction hit the hard 84 C thermal guard
after 12 of 26 frames. Per-frame tensors and manifests are now atomic, partial
resume validates shape/dtype/finiteness, and new frames are thermally paced;
the resumed run reused the 12 sealed tensors and completed all 26 frames with
`benchmark_masks_opened=false` and `label_content_opened=false`. Its first
1.4-million-Gaussian SigLIP load was then killed with exit 137 because the
parent checkpoint tensors and repeated CPU best-state snapshots overlapped in
host memory. The parent tensor copy is now discarded after reconstruction,
best-state refresh is in-place, and lineage hashing precedes multi-GiB loading.
The retry reused every completed upstream artifact and completed all 64
SigLIP steps; the previous run was killed before emitting any step record.
Raw validation rose from 0.790628 to 0.791057, SigLIP rose from 0.778118 to
0.778279, and MPR changed from 0.99946249 to 0.99946046. The sealed field has
entered its source-only region stage with explicit optimizer-state offload.

Orchids has now completed its target-blind generic-response stage and passed
the content-hash-bound Method-v1 gate, the first of nine required scene gates.
Generic validation loss fell from 0.246898 to 0.209225, semantic validation
rose from 0.439117 to 0.453508, and SigLIP validation rose from 0.844733 to
0.844868. Raw and MPR changed by only 0.000011 and 0.000025 respectively,
inside their frozen 0.0002 drop limits. The sealed final field SHA-256 is
`429123de3a9b2aa52ce09ed6cfa718a16f4798479ab48440be3e08ab730d3e9e`.
With GPU0 released, fern construction has started under the same corrected
runner. This remains a pre-GT field gate, not a benchmark score.

Cross-scene execution exposed one further scheduling defect. Fern's
factorized MPR declared a 12.43 GB host-memory peak and held about 9.6 GB RSS
while leaves loaded its 1.4M-Gaussian region field; leaves was killed with exit
137 before step 1 even though both GPUs separately had capacity. All SPIn MPR,
base-field, SigLIP, region, and generic stages now share a cross-process host
memory lock, while source extraction, crop extraction, validation planning,
and method gates may still overlap. The real retry shows leaves holding the
region lock and fern waiting at exact-raw MPR. Fern's completed 3.424 GB
factorized MPR was preserved, and no partial leaves region field was accepted.
With the conflicting MPR removed, leaves completed region step 1 plus full
validation at about 21.6 GB RSS: semantic/raw/SigLIP/MPR validation were
0.451463/0.791054/0.778290/0.999460 respectively. This crosses the previous
failure boundary; the stage remains active through its step-32/64 gates.

That successful step-1 validation exposed a separate optimizer-update peak at
step 2. PyTorch foreach AdamW attempted to materialize a full 2.67 GiB
second-moment square-root denominator for the 1.4M by 512 local-code tensor,
with only 2.32 GiB free. Offloaded moments are now staged through fixed
16,777,216-element GPU chunks (64 MB per float32 moment/denominator tensor),
using the unchanged AdamW formula and CPU state. A three-step fixture is
bitwise equal to PyTorch single-tensor AdamW for parameters, step, exp_avg, and
exp_avg_sq. The first real retry acquired the host-memory lock while fern
waited, proving that the allocator peak was isolated from cross-scene RSS.

The next real retry passed step 1 with the chunked optimizer, then exposed the
matching 2.67 GiB local-code gradient allocation on step-2 backward: the old
`set_to_none=True` policy discarded the already valid giant gradient buffer.
The offloaded path now zeros and reuses that buffer in place; non-offloaded
callers retain `set_to_none`. Pointer reuse/zeroing and the unchanged fallback
are covered by tests. The next real retry is queued behind fern exact-DINO MPR
and must still cross step 2/32/64.

No joint five-benchmark row is eligible yet. Existing historical peak numbers
must not be combined into a virtual incumbent.

## Reproducible artifacts

- Per-scene spatial field: `official_siglip2_spatial_w005_s0_64.pth`
- Per-scene spatial+region field:
  `official_siglip2_spatial_region_w005_s0_64.pth`
- Per-scene frozen-basis generic-response field:
  `generic_text_response_w005_s0_64.pth` (Figurines uses the repaired
  `generic_text_response_w005_s0_64_lineage.pth`)
- Joint-basis early-stop ablation:
  `generic_text_response_basis_w005_lr5e4_s0_64.pth`
- Source-only official spatial bundle:
  `canonical_teacher_features_v2/figurines_source_only_siglip2`
- Source-only genuine crop-summary bundle:
  `optimization_20260716/semantic_teacher_train`
- Frozen global region bridge:
  `global_region_summary_coco15000_full_context_local_scales_imageholdout.pth`
- Frozen Method-v1 primitive caches: `primitive_query_method_v1.pth` under
  each scene directory.
- Frozen LERF2D Method-v1 reports: `lerf2d_eval_method_v1_primitive` under
  each scene directory. The older `lerf2d_eval_method_v1` and Figurines
  `lerf2d_eval_method_v1_lineage` directories are dense-readout diagnostics.
- Frozen LERF3D Method-v1 reports: `lerf3d_eval_method_v1/<scene>/` under each
  scene directory.
- Frozen ScanNet Method-v1 reports: `scannet_vala_method_v1/` under each of
  `optimization_20260815/core_method_v1/scene0000_00/`, `scene0062_00/`,
  `scene0070_00/`, `scene0097_00/`, `scene0140_00/`, `scene0347_00/`,
  `scene0400_00/`, and `scene0590_00/` in the results root.
- Frozen NVOS full-eight Method-v1 result:
  `optimization_20260815/core_method_v1/nvos/method_v1_readout/full8_20260816/method_v1_nvos_full8_results.json`.
- Promoted NVOS development candidate:
  `optimization_20260815/core_method_v1/nvos/method_v1_readout/field_box_signed_points_sam3_candidate_20260816/result.json`.
- SPIn exact-capability shard-width correction:
  `paper/artifacts/spin9_method_v1_capability_shard512_implementation_correction_20260816.json`.
- SPIn capability-cohort v2 metadata correction:
  `paper/artifacts/spin9_method_v1_capability_cohort_v2_metadata_correction_20260816.json`.
- SPIn complete-grid SigLIP AMP correction:
  `paper/artifacts/spin9_method_v1_siglip_global_projection_amp_correction_20260816.json`.
- SPIn region allocator correction:
  `paper/artifacts/spin9_method_v1_region_allocator_correction_20260816.json`.
- SPIn crop-teacher atomic resume/thermal correction:
  `paper/artifacts/spin9_method_v1_crop_teacher_resume_thermal_correction_20260816.json`.
- SPIn large-scene host-memory correction:
  `paper/artifacts/spin9_method_v1_large_scene_host_memory_correction_20260816.json`.
- First complete new SPIn Method-v1 scene gate (orchids):
  `paper/artifacts/spin9_orchids_method_v1_gate_result_20260816.json`.
- SPIn cross-scene host-memory stage lock:
  `paper/artifacts/spin9_host_memory_stage_lock_correction_20260816.json`.
- SPIn large-scene bounded-chunk AdamW correction:
  `paper/artifacts/spin9_large_scene_chunked_adamw_correction_20260816.json`.

## Verification

- The current SigLIP/xFormers, allocator/offload, large-scene snapshot,
  crop-resume, SPIn runner/full-nine barrier, and NVOS readout regression slice
  passes 79/79; the only warning is PyTorch 2.0.1's existing TypedStorage
  deprecation notice.
- The focused Method-v1, primitive-cache, finetune, generic-response,
  semantic-alignment, and ScanNet external-primitive evaluator regression
  suites pass 43/43, including the Waldo one-row alignment, fail-closed larger
  mismatch cases, configured ScanNet frame-authority fallback, and the full
  LERF3D authority formula fixture.
- 104 audited LUDVIG wrapper/full-carrier and SPIn adapter tests pass.
- Python compilation and `git diff --check` pass.
- The NVOS field-box/signed-evidence and SPIn full-nine barrier regression
  slice passes 16/16.
- The frozen five-contract validator still reports zero eligible joint rows;
  it correctly refuses to stitch historical peak numbers into a virtual
  five-benchmark result.
