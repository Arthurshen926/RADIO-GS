# Benchmark design

## Research question

UQIS asks a controlled question: given persistent 3-D scene state, how well can
the same method respond to text, image, 2-D point and 3-D point queries when
every query must produce the same official-mesh output and obey its own input
contract?

It is intended for horizontal comparison and ablation, not as a claim to a
large community benchmark. The current cohort contains nine ScanNet scenes and
67 physical target instances. Each target has four evaluator-private paired
queries, yielding 268 query identities.

## Why a unified output domain

All methods return one finite `float32[V]` vector in `[0,1]`, where `V` is the
number of vertices in the content-bound official ScanNet mesh. The vector order
is fixed by the public scene manifest. A method may internally use Gaussians,
voxels, meshes, neural fields or multiple modality-specific feature fields, but
the evaluator never changes its ground-truth domain.

This avoids four common confounds:

1. text localization being scored by a peak-in-box test while prompts use IoU;
2. image methods receiving target-view RGB that prompt methods cannot access;
3. 2-D methods being evaluated in a favorable camera rather than in 3-D;
4. different methods silently choosing different meshes or sampling density.

## Authorized query interfaces

| Modality | Method-visible query input | Explicitly forbidden |
|---|---|---|
| text | scene ID and one expression | target ID, pairing, RGB, evaluator labels |
| image | scene ID and an opaque normalized 224×224 RGB crop | source frame ID, bbox metadata, pairing |
| point-2D | scene ID, camera matrices, raster size and one positive pixel | captured RGB, target mask, SAM at query time |
| point-3D | scene ID and one world-space point | target identity and labels |

The point-2D raster is rendered from method scene state. It is an interaction
surface, not a captured RGB observation. The 2-D and 3-D prompts correspond to
the same measured surface point, allowing a controlled comparison of raster
interaction versus direct world-space interaction.

## Core and relational challenge

The v0.2 task decomposition separates two capabilities:

- **Unified-Query Core Cohort**: 31 targets with at least one correct,
  class-mentioning, view-independent Nr3D expression explicitly annotated
  `uses_spatial_lang=false`. These targets support the primary four-modality
  comparison.
- **Relational Text Challenge**: 36 targets for which all otherwise eligible
  Nr3D expressions require spatial/relational language. This is text-only and
  never averaged into the primary four-modality score.

The target objects and scenes were not replaced to make text easier. Only the
canonical expression and reporting tier changed. Every one of the nine scenes
contains between two and five Core targets.

Same-class distractors remain useful for instance grounding and are reported
as a diagnostic subset. UQIS does not require every Core expression to encode
same-class relational reasoning; that harder capability lives in the separate
challenge.

## Persistent field accounting

A method declares a per-scene field dependency set for each modality. Shared
artifacts are charged once; distinct CLIP, DINO or task-specific fields are all
charged. A method may therefore be a valid multi-field comparator without
pretending to use one universal representation.

The inventory reports:

- persistent artifact hashes and bytes;
- modality-to-field dependencies;
- whether one universal field is actually used;
- training and query-time device/runtime information;
- any persistent graph topology or calibration artifact.

## What UQIS does not claim

- It is not a replacement for broad open-vocabulary 2-D benchmarks.
- It does not make historical LERF localization percentages comparable to mesh
  AP or IoU.
- Nine scenes are enough for the controlled study, not for a universal claim
  over all ScanNet environments.
- Current candidate results are not formal leaderboard rows until dev
  calibration and external evaluation authority are deployed.
