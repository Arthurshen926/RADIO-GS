# Native semantics and structural membership: controlled negative result

The structural membership prototype is implemented, trained and evaluated on
both development scenes. **Do not promote it.** It preserves the entire parent
field exactly, but its learned membership readout damages final query ranking.
This isolates a second capability-loss location: protecting the representation
alone does not protect the decision made from that representation.

## Relevant prior work and the actual transferable ideas

- [GARField, CVPR 2024](https://www.garfield.studio/) explicitly addresses
  overlapping/conflicting multi-level SAM masks. It learns 3D affinity conditioned
  on physical group scale with same-mask/different-mask contrastive supervision.
  Part and whole need not receive contradictory labels in one unconditional
  grouping space.
- [SAGA](https://jumpat.github.io/SAGA/) learns Gaussian affinity from SAM masks
  with scale-aware contrastive supervision and a soft scale gate. It supports
  2D visual prompts selecting 3D elements. Its mechanism is relevant to future
  SAM-style prompts; it does not establish text generalization for our system.
- [OpenGaussian, NeurIPS 2024](https://3d-aigc.github.io/OpenGaussian/) identifies
  weak point features and inaccurate 2D–3D associations as problems. It learns
  consistent/discriminative instance features, discretizes with a two-level
  codebook, then associates rendered instance masks to CLIP features using IoU
  and feature similarity. Grouping and language association have different jobs.

These works address related grouping/association problems. They do not prove
our exact SigLIP projection-forgetting diagnosis or supply a guarantee that our
particular compact field preserves every native teacher capability. This trial
is inspired by the responsibility separation, not a reproduction of these works.

## Implemented controlled experiment

Parent runs: `partition96_protected_2106` (figurines),
`partition96_protected_3240` (ramen). Freeze all parent parameters and buffers:
64 semantic coordinates, 32 structure coordinates, decoder, query adapters,
and calibration. The new shared text/image head is the only optimized module.

At query time native cosine selects up to four observed elements, with a
scene-normalized geometry exclusion radius of 0.02. Each element receives
seed-relative structural absolute differences/products, distance, native cosine,
seed-relative cosine and a modality bit. A 64-unit ReLU MLP predicts membership
for each seed; maximum over seeds produces the element response. No label or
reference-target identity selects these seeds. Geometry is bound at cold load;
no extra per-element semantic state or training cache is needed at inference.

The matched semantic-only control zeros structural differences/products and
distance. It has the same network capacity and seed policy. Thus this isolates
**structural/geometric inputs to the head**, not every use of geometry: the
control also uses geometry to select diverse seeds.

Both arms use the existing source-only signed supervision: grounded text,
image cross-view tasks and source-region image queries, conflict exclusions,
unknown abstention, and joint element/normalized-rendered pixel BCE+Dice.
Each receives 1,404 steps in figurines or 2,160 in ramen, two full image-task
passes and cyclic text tasks. Text/image share the head. Prompt inference is
unchanged. Training never opens benchmark labels or target RGB.

This head is **not scale conditioned**. Its normalized distance is not a physical
mask-size hierarchy. Existing suffix teacher reconstruction and contour training
also do not establish scale-consistent instance affinity. These are remaining
architectural limitations, not demonstrated causal explanations for every error.

## Fixed-protocol results

All values below are percentages, reported as IoU / AP. Full text has 56
figurines and 71 ramen observations. Reference image has 35 and 57 tasks;
its labels define category unions, not verified physical instances.

| Scene / query | Protected parent | Structural head | Same-capacity semantic-only head |
|---|---:|---:|---:|
| figurines text | 34.63 / 57.64 | 26.68 / 41.42 | 0.00 / 56.53 |
| ramen text | 2.14 / 22.72 | 10.26 / 18.06 | 0.00 / 22.63 |
| figurines reference image | 33.42 / 54.82 | 29.84 / 44.57 | 8.97 / 53.97 |
| ramen reference image | 13.13 / 34.93 | 17.92 / 29.05 | 20.19 / 34.94 |

Image parent uses the previously completed source-balanced scalar calibration,
not the degenerate uncalibrated all-negative result. Text parent is unchanged.
No new threshold search, target-label fitting or teacher inference on target
RGB was performed. Image and full-text cohorts must not be averaged together.

The semantic-only head largely preserves AP while failing text thresholding;
adding structure reduces AP in both scenes and both query modes. Therefore the
structural head's occasional IoU gains are insufficient evidence of improved
semantic retrieval. For example, figurines miffy IoU drops 36.62→4.58, pikachu
74.95→36.86, porcelain hand 52.16→0. Meanwhile green apple rises 33.11→60.87.
This is selective capability loss, not a globally broken renderer.

Source automatic-seed evidence is incomplete: at least one confirmed-positive
seed occurs in 19.05%/13.13% of text episodes and 64.39%/45.56% of image episodes
(figurines/ramen). However 76.19%/82.83% of text episodes have **no labelled seed**.
These are partial, view-restricted labels: unlabelled seeds are not demonstrated
wrong seeds. Do not interpret these numbers as true seed precision or recall.

## Integrity and acceptance

All four runs verify:

- Every original parent state tensor exactly unchanged.
- Native cosine maximum error 0; cold-load output maximum error 0.
- Same frozen geometry and observed domain. Prompt metrics exactly unchanged.
- Source artifact identities and source query/target frame checks.
- Model/trainer implementation hashes unchanged during fitting.

Four new behavioral tests and 16 existing field/decoder tests pass. They check
full-map seed selection under row slicing and query batching, unsupported-element
exclusion, actual structural dependence, semantic-control invariance, frozen
optimizer ownership, and missing bound geometry rejection. Cold-load equivalence
is also tested on each real exported field.

The acceptance report rejects both heads. These checks are local development
comparison gates, not sufficient criteria for independent benchmark promotion.
Exact preservation is relative to the **existing source-fused 64D anchor**;
it does not recover losses already introduced by native compression/fusion.

## Next core correction, constrained by this result

Do not tune this unrestricted MLP's depth, threshold or loss weight to rescue it.
The next implementation should learn **query-independent multi-scale grouping**
from reliable masks, then associate each candidate group with a semantic vector
computed through the frozen native decoder. Text/image select candidate groups
using the fixed native similarity operation. Structural learning determines
which elements form a group; it cannot learn a replacement open-vocabulary
identity metric from 14 source concepts.

Before another full training run, measure whether candidate groups cover true
objects/parts and whether their frozen native descriptors preserve query ranking.
These diagnostic label measurements must remain separate from source-only fitting.
If candidate membership already misses thin parts or mixes instances, a semantic
association head cannot repair the missing carrier. Group scale selection and
query type (category union versus reference instance) must be explicit; one
unconditional binary target does not solve this ambiguity.

This next architecture is not implemented or validated by the present trial.
ScanNet-OVS, NVOS and SAM-style 2D prompt performance remain unverified.

## Reproduction and artifacts

- Trainer: `scripts/audits/train_structural_membership.py`.
- Readout: `radio_gs/models/structural_membership_readout.py`.
- Exports: scene directories under
  `/mnt/pool/sqy/results/RADIO-GS/output/component_retrieval_pipeline_20260915`,
  runs `structural_membership_v1` and `structural_membership_v1_semantic_only`.
- Evaluation: existing `evaluate_frozen_field_lerf_labels.py` and
  `evaluate_real_reference_queries.py`, width 480 and threshold 0.5 unchanged.
- Detailed results: `paper/artifacts/{scene}_structural_membership_{text,reference}_20260916.json`.
- Hashed comparisons and rejection gates:
  `paper/artifacts/structural_membership_acceptance_20260916.json`.
- Summary generator: `scripts/audits/summarize_structural_membership.py`.
