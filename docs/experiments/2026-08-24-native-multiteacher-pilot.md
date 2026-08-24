# Native multi-teacher pilot

## Outcome

This round validates the architectural change in a narrower form than the
initial proposal.  RADIO remains a useful shared anchor, but correct-level
native object statistics add information that the compact RADIO capability
does not contain.  The successful intervention is the ScanNet categorical
region residual.  Naively updating the shared L512 code with native DINO and
adding native DINO to LERF proposal identity both fail disjoint-source gates.

The evidence therefore supports

`RADIO anchor + variable-aligned native teacher + one compact field`,

not an undifferentiated sum of RADIO, DINO, SAM and SigLIP reconstruction
losses.

## Coverage of the external multi-teacher proposal

The proposal is not yet fully implemented.  Its recommendations now have the
following evidence boundary:

| recommendation | status and evidence |
|---|---|
| RADIO as shared anchor rather than sole sufficient statistic | implemented; frozen-L512 A wins all three DINO sentinels |
| native DINO before exact MPR | implemented in matched A/B/C; latent-updating B/C rejected |
| native SigLIP category capability before exact MPR | implemented and promoted on ScanNet paper8 |
| native SAM extent plus native SigLIP region identity | implemented; exact sidecar improves all 24 ScanNet mIoU comparisons |
| compact the native region teacher into shared L512 | implemented; positive paper8 macro, but scene0400 and teacher gap remain |
| source-heldout complete object-membership reconstruction | implemented on LERF; current proposal carrier fails the fixed gate |
| native DINO/SigLIP object association for LERF | implemented and rejected by disjoint-source gates |
| registered prompt identity plus native SAM memory | implemented on NVOS full8; strong on 5/8 scenes, unconditional propagation rejected |
| one joint L512 optimization over RADIO+DINO+SAM+SigLIP | not implemented; available A/B/C evidence argues against a naive loss sum |
| independent complete physical-instance/track authority for LERF | not available; this is the current high-value data/teacher gap |

Consequently, the external proposal has materially changed the ScanNet and
NVOS compilers and has falsified several LERF shortcuts.  It has not yet
established a single jointly trained all-teacher field or a jointly improving
LERF2D/3D instance posterior.

## Matched native-DINO A/B/C sentinel

All arms use the same L512 capacity, Gaussian geometry, source observations,
random decoder initialization, minibatch schedule and 800 decoder steps.

- A freezes L512 and trains one global native-DINO decoder.
- B updates L512 with native DINO plus a RADIO preservation loss.
- C updates L512 with native DINO only.

The metric is mean cosine on disjoint source-heldout exact-MPR observations.

| scene | A frozen | B RADIO-anchored | C native-only | B RADIO cosine | C RADIO cosine |
|---|---:|---:|---:|---:|---:|
| LERF Figurines | 0.75675 | 0.74755 | 0.74753 | 0.99984 | 0.99984 |
| ScanNet scene0000 | 0.71348 | 0.70400 | 0.70396 | 0.99965 | 0.99961 |
| NVOS Fern | 0.86797 | 0.86673 | 0.86673 | 1.00000 | 1.00000 |

All three sentinels select A.  B/C preserve RADIO but fit the source-training
cohort at the expense of held-out native DINO.  A full DINO-driven field
reconstruction is therefore rejected.  This is a DINO-only representation
sentinel, not a claim that object-level native SAM or SigLIP supervision is
unnecessary.

## ScanNet native region residual

Official query-free source SAM3 masks supply region extent.  Independent native
SigLIP2 masked and expanded-context crops supply category identity.  Evidence
is averaged once per source view and then fused as one centered residual into
the already promoted L512 capability-before-MPR score.  Wall, floor and ceiling
replay the primitive score.  There are no class-, scene- or Gaussian-indexed
parameters.

The fixed `alpha=0.25`, two-view minimum and 0.5 view-agreement rule were chosen
on scene0000/0062/0070/0097.  Without further selection, the same rule improves
the independent scene0140/0347/0400/0590 cohort:

| cohort | split | delta mIoU | delta mAcc |
|---|---:|---:|---:|
| development4 | 19 | +0.01208 | +0.00638 |
|  | 15 | +0.01271 | +0.00861 |
|  | 10 | +0.01082 | +0.00700 |
| confirmation4 | 19 | +0.00825 | +0.00281 |
|  | 15 | +0.00731 | +0.00363 |
|  | 10 | +0.00742 | +0.00207 |

On paper8, the current deployable baseline and new compiler are:

| metric | split19 | split15 | split10 |
|---|---:|---:|---:|
| baseline mIoU | 0.35613 | 0.35300 | 0.45954 |
| native-region mIoU | **0.36630** | **0.36301** | **0.46866** |
| delta mIoU | +0.01017 | +0.01001 | +0.00912 |
| delta mAcc | +0.00460 | +0.00612 | +0.00453 |

All 24 scene-by-split mIoU comparisons improve.  Twenty-two of 24 mAcc
comparisons improve; scene0140 split19 and scene0400 split10 regress by about
0.002.  The paper8 macro and fixed confirmation gates pass, but per-scene
Pareto dominance is not claimed.  The compiler is promoted as development
method evidence, not as an SOTA claim.

### Frozen-L512 distillation of the region teacher

The exact native sidecar was subsequently distilled into a scene-global
residual decoder over the existing frozen L512 row.  This adds no Gaussian
state.  The fixed 400-step decoder produces paper8 mIoU
`0.36027/0.35957/0.46626`, gains of
`+0.00414/+0.00658/+0.00672` over the restored baseline.  Seven of eight
scenes improve on every split; scene0400 regresses.  The corresponding mAcc
gains are `+0.00360/+0.00552/+0.00477`.

This is a positive compact-field result, but it is weaker than exact native
region evidence (`0.36630/0.36301/0.46866`).  Cosine reconstruction of the
1536-D descriptor does not fully preserve the small categorical query margins.
The next compact decoder must therefore gate the query-variable response, not
only descriptor cosine.  The current student is not claimed to have fully
absorbed the native sidecar.

## LERF native proposal and membership gates

Native DINO proposal descriptors were built by pooling official DINOv2 tokens
inside the exact 277-proposal source-SAM cohort.  A source-only cosine gate of
0.60 separates the upper associated-pair tail from nonedges.  Enforcing that
authority also uncovered and fixed a real implementation error: SAM parent
ascent could previously replace a DINO-accepted child with a parent below the
appearance threshold.

The corrected parent guard is logically valid but does not improve the matched
benchmark method.  Figurines reaches LERF2D mIoU 0.40943 and LERF3D mIoU
0.52127, below the corresponding fallback controls.  It is not promoted.

The direct source-heldout object gate is also negative:

| source-heldout method | macro IoU | delta over primitive |
|---|---:|---:|
| primitive SigLIP similarity | 0.12457 | -- |
| frozen-L512 native SigLIP membership decoder | 0.13913 | +0.01456 |
| frozen-L512 native SigLIP + DINO decoder | 0.13557 | +0.01099 |
| training-view proposal retrieval selected on calibration residue | 0.08783 | -0.01533 vs same-topk SigLIP |

The multimodal decoder fails the preregistered 0.20 absolute and +0.02 gain
gate.  The proposal-retrieval experiment uses candidate views 0/1, selects its
descriptor and top-k on residue 2, and evaluates residue 3.  Calibration picks
SigLIP+DINO top-2 at 0.13837, but evaluation falls to 0.08783.  Native DINO
therefore helps neither learned membership nor stable cross-view object
transport on this proposal carrier.

This is the central LERF diagnosis: the remaining problem is not dense feature
capacity or the absence of another appearance descriptor.  The source-SAM
proposal/membership carrier does not form stable complete 3D instances across
views.  Another benchmark posterior sweep has low expected value until that
authority changes.

An official SAM3 video-tracker control now rules out a tempting shortcut.  A
first pass over the eight exact-MPR anchors was not interpretable because the
anchors skip about ten captured frames.  The corrected experiment inserts
every legal registered source RGB between anchors and evaluates only the same
held-out exact anchors.  Three disjoint early/middle/late windows still fail:

| dense source window | independent-SAM extent IoU | native-DINO seed cosine |
|---|---:|---:|
| early | 0.25237 | 0.55228 |
| middle | 0.09469 | 0.25094 |
| late | 0.21354 | 0.24091 |

The fixed source gate is 0.50 extent and 0.60 identity.  Automatic long-range
video propagation therefore cannot replace explicit 3D object membership on
this camera path.  The result does not reject official SAM3 masks themselves;
it rejects treating one automatically seeded 2D track as stable 3D instance
truth.

## NVOS and SPIn boundary

NVOS Fern independently reproduces the DINO A/B/C conclusion without opening a
benchmark query or target mask.  The benchmark method remains registered field
identity plus protocol-authorized RGB and official-SAM extent.  The existing
native all-view compiler is complete at 0.82624 macro IoU with its frozen local
identity limiter; this round does not claim a new NVOS metric.

A new official-SAM3 video-memory control uses the official positive/negative
scribble directly and follows the shorter cyclic registered-RGB path.  It
improves Fern, both Horns, Leaves and Trex, including `+0.27720/+0.26450` on
Horns-center/left and `+0.31223` on Leaves.  But point-only initialization
under-segments Flower, Fortress and Orchids, so its full8 macro is `0.81123`,
below the same-chain target-only `0.81743`.  Unconditional propagation is
rejected.  This sharp split supports the identity--extent theory while showing
that prompt-side complete-instance proposal selection and a target-blind
reliability authority are mandatory.

The complete prompt-proposal version closes that gap.  Official SAM3 first
generates 200 prompt-view regions; the full positive/negative scribble chooses
the identity-consistent complete extent, and its box plus signed points seed
video memory.  Raw full8 rises to `0.90547`.  Reusing the already frozen NVOS
region reliability rule (`target SAM score >= 0.5 OR field overlap < 0.5`)
authorizes video memory on Flower, both Horns, Leaves and Trex and retains the
sealed target-only prediction on Fern, Fortress and Orchids.  The final
development result is `0.92555` macro IoU and `0.98553` pixel accuracy, versus
`0.81762` target-only (`+0.10793`), with no scene regression.  It is independent
of LUDVIG and opens no target mask during prediction, but remains explicitly
target-RGB-assisted and is not an outcome-blind paper/SOTA result.

SPIn has been removed from the short-term experimental critical path because
the available benchmark cohort is incomplete (9/10).  No SPIn process is
running.  Existing fields, intermediate artifacts and completed gates are
preserved, but all current compute is assigned to LERF2D/3D, ScanNet OVS and
NVOS.

## Implementation corrections

- The ScanNet native teacher tolerates a legal source frame with zero SAM
  proposals instead of stacking an empty tensor.
- Topology-free score-cache export is explicit and is allowed only in
  score-cache-only mode when a separately hash-bound native topology is used.
- Scene materialization uses a file lock, preventing two GPU queues from
  writing the same score cache.
- LERF membership evaluation caches the proposal query encoding and frozen
  Gaussian encoding.  The primitive baseline can stage its normalized feature
  table once on device.  These changes preserve dense numerical equivalence
  while removing repeated high-dimensional work.
- Source proposal retrieval caches sparse visibility/support sets instead of
  allocating a full carrier boolean table for every candidate.

## Bound results

- ScanNet aggregate:
  `paper/artifacts/scannet_native_region_residual_paper8_result_20260824.json`
- ScanNet frozen-L512 region distillation:
  `paper/artifacts/scannet_native_region_distilled_l512_paper8_result_20260824.json`
- DINO A/B/C summaries:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260824/native_multiteacher_v1/{figurines,scene0000_00,nvos_fern}/native_dinov2_abc_matched_v2/abc_summary.json`
- LERF multimodal membership:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260824/native_multiteacher_v1/figurines32/source_membership_gate/native_siglip2_dinov2_frozen_l512.pt.json`
- LERF disjoint-view retrieval:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260824/native_multiteacher_v1/figurines32/source_membership_gate/native_proposal_retrieval_gate.json`
- NVOS prompt-proposal video reliability full8:
  `paper/artifacts/nvos_sam3_prompt_proposal_video_reliability_full8_result_20260824.json`
