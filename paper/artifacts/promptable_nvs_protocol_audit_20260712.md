# NVOS / SPIn-NeRF promptable segmentation protocol freeze

This note freezes the evaluation contract before producing any GaussFM number.
It prevents paper-reported baselines, query-time SAM systems, and reusable
feature fields from being silently treated as the same capability.

## Outcome

NVOS and SPIn-NeRF share the high-level task “one reference-view prompt,
novel-view binary masks,” but they must remain separate benchmarks:

- NVOS uses fixed positive/negative scribbles and one unseen target per task.
  The strict GaussFM run excludes the target RGB from field training and query.
  Sparse points carrying any target-camera observation are also removed before
  3DGS initialization. Camera poses/intrinsics still come from the upstream
  joint reconstruction; this shared-calibration exception is reported rather
  than described as fully target-pixel-independent geometry.
- The original SPIn-NeRF task starts from sparse positive/negative clicks on
  one source view. The public annotation release contains complete masks, not
  a frozen click list. Therefore the complete-first-mask GaussFM readout is a
  separately named **full-mask support diagnostic**, not an exact SAGA row.
  It allows the scene RGB/poses and evaluates every subsequent annotated
  frame; the first view is input and is not a scored target.

No combined NVOS+SPIn overall is permitted. Foreground IoU is the primary
metric; binary pixel accuracy is secondary because background dominates it.

## Frozen aggregation

NVOS consists of eight equally weighted object tasks: fern, flower, fortress,
horns_center, horns_left, leaves, orchids, and trex. Each task contributes its
single target IoU and accuracy once.

SPIn-NeRF consists of ten equally weighted scenes: orchids, leaves, fern, room,
horns, fortress, fork, pinecone, truck, and lego. First average all non-reference
annotated frames within a scene, then average the ten scene scores. The reference
frame is never counted. Masks are compared at GT resolution; the only permitted
mask resampling is nearest-neighbor.

GaussFM predictions are continuous cosine margins, thresholded with
`score >= 0` fixed before evaluation. A reproduced baseline that exports binary
masks must declare `binary_mask` and is scored directly (with 0 as background);
it cannot inherit the cosine-margin threshold metadata.

Target masks are scoring-only. They may not choose a checkpoint, feature
prototype, threshold, connected-component rule, mask area, or SAM candidate.
Although reference-only calibration could define a separate full-mask
diagnostic, it is **not used here**: both locked GaussFM manifests use the fixed
zero cosine-margin threshold. In the point-prompt track, the reference mask may
only instantiate the frozen prompt sampler; it may not tune the threshold.
NVOS parameters must be frozen outside the eight target masks.

## Prompt-budget non-equivalence

The final SAGA supplement states that NVOS prompts are a random selection of
positive and negative points from the official scribbles. For SPIn-NeRF it
randomly samples a subset inside and outside the reference mask. It does not
disclose the subset size or random seed. SAGA's published 92.6/98.6 and
93.4/99.2 values therefore cannot be certified as the output of our locked
prompt generator, even if the masks and metric code match.

Consequently, the current GaussFM full-scribble/full-reference-mask prototype
results must not be placed in a table headed “same protocol as SAGA.” Published
numbers remain protocol-audited context. A strict point-prompt main table needs
one deterministic positive/negative point budget and seed frozen first, then
all methods must be rerun under it. This is not target-set calibration: the
reference mask may generate the declared prompts, while every non-reference
mask remains scoring-only.

## Capability blocks

The reusable-feature-field block contains GaussFM, SAGA, and OmniSeg3D. The
field is trained before the object query; at query time it reads only the
reference prompt, produces one fixed 3D selection, and does not run a 2D model
on target RGB.

The online multiview-propagation block contains SA3D-GS, SA-GS/SAGD, and
LUDVIG-SAM/SAM2. These are relevant end-to-end comparisons, but they invoke SAM
on additional RGB views and/or optimize a prompt-specific representation. Their
accuracy does not by itself establish reusable feature-field quality.

SA-GS here means Hu et al.'s segmentation method (now named SAGD), not the
unrelated anti-aliasing or large-scene methods with the same acronym. Its
published SPIn-NeRF table omits Leaves and is therefore a nine-scene context
row, not a valid ten-scene leaderboard row.

## Published context policy

Values copied from final papers are labelled “paper-reported, protocol-audited
context”; they are not represented as local reproductions, output from the
same evaluator, or exact-prompt comparisons. The exact values, sources, cohort
exceptions, and capability class are machine-readable in
`promptable_nvs_protocol_registry.yaml`.

Two stale values are explicitly banned: the early SAGA project-PDF results
(NVOS 90.9 / SPIn 88.0) and OmniSeg3D arXiv-v1 SPIn 95.2. Use the final
published values in the registry.

LUDVIG's NVOS result is context-only even at the task level: its released
uplifting code visits all COLMAP views and therefore sees the target RGB. This
does not use target GT, but it differs from the strict unseen-target split.

## GaussFM reporting contract

The main GaussFM row is the rendered RADIO `sam3` adaptor feature field with a
reference-only readout and no target-view RGB decoder/refiner. It must not be
described as the official SAM/SAM2/SAM3 decoder. Any optional RGB SAM-assisted
diagnostic belongs in the online block and must disclose target RGB use.

Every exported run must retain the dataset manifest, protocol hash, source and
target frame IDs, field checkpoint hash, fixed readout parameters, per-frame
metrics, per-scene metrics, and final macro. A run is ineligible if any target
mask was opened before the final scoring phase.
