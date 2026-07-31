# Canonical feature/query recovery iteration — 2026-07-31

## Status

This iteration is diagnostic infrastructure and method development, not a
main-table promotion. `canonical-mpr-v3` remains the last promoted core and the
2026-07-24 v4 declaration remains a candidate.

The strict historical HCD row (`0.3761` LERF mIoU) is the relevant architecture
comparison. The old `0.5889` paper route also used threshold/temperature
selection, component cleanup, and boundary post-processing and is not directly
comparable with the canonical query-invariant route.

## Audited implementation

The promoted representation is a compact per-primitive field:

```text
frozen 3DGS
  -> C-RADIO spatial observations
  -> canonical MPR
  -> local-128 + reliability-conditioned primitive fusion
  -> affine coefficient-256
  -> affine RADIO-1280 decoder
  -> frozen DINOv3/SAM3 capability views
  -> SurfaceRegion text summary readout
  -> typed query unary
  -> optional support graph
  -> scalar/primitive readout
```

Two implementation facts matter:

1. The promoted architecture has `coarse_dim=0`. Its reliability channels were
   effectively `[view coverage, valid, valid]`, so it does not contain the old
   hybrid model's spatial-content branch.
2. The frozen NVOS closeout runner trains the canonical primitive field and
   immediately builds capability views; it does not invoke the declared
   exact-render replay fine-tuner. The intended mainline description and this
   benchmark execution path therefore differ.

Controlled historical ablation assigns most of the real capacity loss to the
nonlinear HCD decoder: full HCD `0.4850`, no refiner `0.4796`, no hybrid branch
`0.5069`, and direct projection `0.2596`. The spatial hash alone is not the
missing capability. The dominant losses are nonlinear decoding/context and
primitive-to-pixel observation.

## NVOS diagnosis and retained changes

The frozen eight-scene chain was:

| stage | macro foreground IoU |
|---|---:|
| unary | 0.740130 |
| propagated | 0.749029 |
| seeded connected component | 0.726404 |

The evaluator had four independent problems:

- official prompts were downsampled to roughly `46 x 62` before lifting;
- legacy lifting used a footprint/depth proxy instead of the exact
  front-to-back raster adjoint;
- prompt mass built prototypes/seeds but was not retained as a typed direct
  unary;
- `SEEDED_COMPONENT` was used for a full `REGION`, destroying valid
  multi-component foreground (especially Orchids).

Retained implementation:

- joint-mass registered unary
  `(m_pos - m_neg) / max_i(m_pos + m_neg)`;
- exact raster-adjoint prompt registration and independent scalar-render
  resolution;
- explicit `unary_prior`, `propagated`, and `connected` readout stages;
- `ALL_COMPONENTS`/continuous readout for region prompts;
- valid-conditioned scalar rendering with separate semantic coverage;
- a coverage-power abstention family
  `E[p | valid] * P(valid)^beta`, where `beta=0` is pure conditional scoring
  and `beta=1` exactly recovers total-alpha dilution;
- memory-mapped loading/compaction for multi-GB dense capability archives.

The direct unary is added after feature-reliability shrinkage because a
registered observation must not be erased by uncertainty in the semantic
field. Frozen behavior remains available with weight zero.

## NVOS diagnostics

All rows below use the same strict protocol and no target calibration. The
half-resolution registration / 8x scalar render setting is a diagnostic
candidate; it was inspected on benchmark scenes and is not promotion-eligible
until frozen on a disjoint development split.

| scene / variant | unary | propagated/final |
|---|---:|---:|
| Orchids frozen | 0.700985 | 0.700091 / connected 0.482042 |
| Orchids exact + high render + valid conditional | 0.736426 | 0.726721 |
| Orchids same, direct unary disabled | 0.724007 | 0.725052 |
| Orchids exact + high render + total alpha | 0.587117 | 0.575802 |
| Orchids legacy registration + high render + valid conditional | 0.705542 | 0.694838 |
| Orchids exact + low render + valid conditional | 0.700589 | 0.697457 |
| Fern frozen | 0.766102 | 0.764152 / connected 0.742245 |
| Fern exact + high render + valid conditional | 0.763406 | 0.763741 |
| Flower frozen | 0.851562 | 0.850562 / connected 0.838188 |
| Flower exact + high render + valid conditional | 0.851244 | 0.847004 |
| Fortress frozen | 0.928827 | 0.928640 / connected 0.929832 |
| Fortress exact + high render + valid conditional | 0.858978 | 0.860112 |
| Horns-center frozen | 0.759949 | 0.777457 / connected 0.776406 |
| Horns-center exact + high render + valid conditional | 0.702346 | 0.717738 |
| Horns-left frozen | 0.687734 | 0.720563 / connected 0.742355 |
| Horns-left exact + high render + valid conditional | 0.523847 | 0.562251 |

Conclusions:

- Removing hard connected-component selection is a robust semantic correction.
- Exact registration and higher scalar resolution materially help Orchids.
- Pure valid-only normalization is not a global solution: it amplifies
  low-coverage false support in Fortress and both Horns prompts.
- Extending prompt evidence directly to capability-invalid rows improved
  Orchids total-alpha `0.5758 -> 0.6082` but reduced Fortress to `0.7651`.
  That branch was rejected and removed. A single-view raster tail cannot safely
  grant cross-view semantics to an otherwise unsupported primitive.
- The stopped queue has no eight-scene candidate macro; no partial macro should
  be reported as a replacement result.

The correct next step is to select the coverage-abstention rule on a disjoint
registered-prompt development split, then run the eight frozen scenes once.

## Text-query diagnosis and retained changes

The v2 region cache stopped Dijkstra after 256 candidates while also emitting
at most 256 tokens. Consequently, the later core/context selector usually had
no context to select. On the existing training cache:

- 77.1% of 0.45 m regions had no context token;
- 92.2% of 0.70 m regions had no context token;
- 0.45 m and 0.70 m token sets were almost identical in Ramen, Teatime, and
  Waldo Kitchen.

Training used true voxel coverage/agreement while canonical inference exposed
mostly coverage plus validity. V2 then consumed reliability as geometry and
again as a log-attention prior, magnifying the shift.

Retained implementation:

- new region caches use 1024 candidates, radial core/context stratification,
  and still emit at most 256 tokens;
- a matched `uniform_valid` reliability contract is available until rebuilt
  canonical fields expose real agreement;
- the V2 readout can use reliability as input only, avoiding the duplicated
  log prior while preserving old checkpoint digests by default;
- completion-aware SurfaceRegion caches preserve `primary_valid`,
  reconstruction confidence, and primary context; a fallback center sees
  primary context plus itself, not other fallback rows;
- completion routing supports raw-cosine-relative primary peaks instead of the
  invalid fixed `0.5` boundary;
- scalar text rendering supports valid-mass normalization and coverage-power
  abstention, with a reproducible queue switch.

The planned four-scene LERF render gate could not run after the NVIDIA UVM
driver entered an uninterruptible cleanup state on physical GPU1. No GPU reset
or switch to the other user's GPU was attempted. Therefore the retained text
changes are unit-tested infrastructure, not a claimed LERF improvement.

## Canonical capacity recovery for the next field

Retained opt-in training infrastructure:

- raster MPR can record mean-resultant directional agreement instead of
  `[coverage, valid, valid]`; legacy reliability remains the default;
- primitive fusion can add token-wise zero-initialized coefficient residual
  blocks; zero blocks exactly preserves old checkpoints;
- the existing primitive-position spatial hash can be enabled as a real coarse
  content branch without introducing screen or batch dependence.

A first v5 capacity experiment should use:

```text
normalized raster observations
mean-resultant reliability
local_dim=128
coarse_dim=64
primitive-position spatial hash
hidden_dim=512
fusion_residual_blocks=2
50-100 primitive epochs
5k-20k exact-render replay steps
```

Selection must happen on query-free held-out raw/DINO/SAM fidelity, relation
preservation, boundary lower-tail retention, and disjoint generic text-response
ranking. Point/render consistency remains intact because every nonlinear
feature transformation happens once per primitive before affine splatting;
text/SAM decisions are compiled to scalar primitive scores before rendering.

## Verification

Targeted canonical-field, MPR, capability-cache, query compiler, NVOS,
SurfaceRegion, text-cache, and LERF readout tests: `155 passed`.

Rejected experimental outputs remain under
`output/optimization_20260731/` for audit only and are not method artifacts.
