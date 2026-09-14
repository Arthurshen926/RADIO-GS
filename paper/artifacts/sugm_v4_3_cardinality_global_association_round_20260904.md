# SUGM v4.3 cardinality and global-association round (2026-09-04)

## Scope

This round tested two explicit hypotheses without changing the frozen surface
carrier, LERF source observations, region-level language keys, query text, or
benchmark thresholds:

1. replace the scene-wide categorical completion posterior with an
   object-balanced, cardinality-invariant independent Bernoulli posterior;
2. reduce fragment duplication with source-only cross-view global object
   consolidation.

All LERF completion construction was query-free and did not read benchmark
labels, held-out RGB, or complete target membership.  Benchmark labels were
opened only by the independent development evaluators.

## Independent Bernoulli completion

Scene-disjoint ScanNet/PFIR validation used 12 training and four validation
scenes, 4,000 steps, the frozen F71 input, and coherent fragment observation
noise at validation time.

| Model | 3D soft mIoU | held-out 2D soft mIoU | unknown coverage | unknown assignment precision | background false-positive mean |
|---|---:|---:|---:|---:|---:|
| categorical, noise matched | 0.48499 | 0.26996 | 0.71117 | 0.70184 | 0.15781 |
| Bernoulli, clean train | 0.44658 | 0.23842 | 0.64250 | 0.69024 | 0.12937 |
| Bernoulli, noise matched | 0.41954 | 0.26666 | 0.76766 | 0.61482 | 0.22712 |

The Bernoulli model increased coverage but converted the gain into false
positive extent.  It did not beat the categorical model.

Artifacts:

- `/mnt/pool/sqy/results/RADIO-GS/output/v4_2_bernoulli_fragment_noise_completion_v2_20260904`
- matched checkpoint SHA256:
  `2e26c7c42144a033899823d26310e9b7a8769261d0567440d45250482f4a964d`

On the four LERF development scenes, the best common rank-union diagnostic
decreased from 0.15820 (categorical completion) to 0.14890 (Bernoulli
completion).  Generic all-token mIoU was 0.07412, versus 0.07723 observed-only
and 0.06892 categorical-completed.  Independent probabilities alone therefore
do not resolve the real 75--159-active-hypothesis deployment mismatch.

Artifacts:

- `/mnt/pool/sqy/results/RADIO-GS/output/v4_2_object_hypothesis_bernoulli_completion_20260904`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_2_object_hypothesis_text_bernoulli_completion_20260904`

## Source-only global consolidation

The previous all-fragment construction retained 260--376 hypotheses from
409--671 fragments.  A fixed cross-view-supported variant reduced this to
37--97 hypotheses.  A visibility-conflict-aware variant retained 93--207.

| Construction | raw stable single ceiling | raw stable top-3 ceiling | canonical stable single ceiling | generic all-token mIoU |
|---|---:|---:|---:|---:|
| previous all-fragment | 0.25623 | 0.28908 | 0.22817 | 0.09556 |
| support-3 union | 0.19704 | 0.20868 | 0.14257 | 0.08774 |
| conflict support-2 union | 0.20604 | 0.22979 | 0.14335 | not promoted past ceiling |
| conflict support-2 medoid | 0.17565 | 0.20969 | 0.14744 | not promoted past ceiling |

The support-3 variant raised all-token cosine mIoU from roughly 0.01 to 0.05291,
which confirms that duplicate-hypothesis dilution is real.  However, every
consolidated variant reduced the object-extent ceiling: probabilistic union
amplified incorrect merge edges, while a single medoid discarded necessary
multi-view extent.  Neither is a valid replacement for the existing default.

Artifacts:

- `/mnt/pool/sqy/results/RADIO-GS/output/v4_3_object_hypotheses_cross_view_support3_20260904`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_3_object_hypothesis_ceiling_cross_view_support3_20260904`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_3_object_hypotheses_conflict_support2_20260904`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_3_object_hypothesis_ceiling_conflict_support2_20260904`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_3_object_hypotheses_conflict_support2_medoid_20260904`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_3_object_hypothesis_ceiling_conflict_support2_medoid_20260904`

## Learned fragment-edge association

A shared eight-scalar MLP was trained on 12 ScanNet scenes and calibrated on
four scene-disjoint validation scenes.  Instance ids were supervision only;
they were not model inputs.  The first clean-oracle-fragment checkpoint reached
pooled AP 0.68364.  Its validation-only F0.5 operating point had precision
1.0 and recall 0.58333.

The first deployment exposed and fixed an important probability-contract bug:
the calibrated binary merge-edge probability had incorrectly also been used as
the categorical fragment-to-hypothesis assignment logit.  Across hundreds of
hypotheses this raised mean null probability to 0.50--0.85 and collapsed the
raw stable-single ceiling to 0.11170.  The repaired implementation uses learned
probability only for grouping and the fixed geometry/appearance/spatial score
for assignment.  Mean null returned to 0.07--0.11.

| Association | raw single | raw top-3 | raw LOO single | canonical single | canonical top-3 | best frozen text rank-union top-3 | generic consensus all-token |
|---|---:|---:|---:|---:|---:|---:|---:|
| coupled edge/assignment (bug) | 0.11170 | 0.12241 | 0.06864 | 0.09847 | 0.10938 | not promoted | not promoted |
| clean learned grouping + fixed assignment | 0.23734 | 0.27136 | 0.15345 | 0.16818 | 0.19367 | 0.16960 | 0.09300 |
| SAM-like learned grouping + fixed assignment | 0.23735 | 0.27016 | 0.15400 | 0.17120 | 0.19860 | 0.15728 | 0.09358 |

The 0.16960 number is a bounded rank-union diagnostic, not the primary
all-hypothesis result.  The primary all-token score did not exceed the existing
0.09556.  The clean model's apparent top-3 gain also did not persist after
noise matching.

The second training cache deterministically generated whole and part regions,
15% proposal dropout, and a bounded subset of light neighbour contamination.
Training batches were scene-equal, object/object-pair-equal, and class
balanced.  Its validation AP was 0.58476; the frozen operating point had
precision 0.96907, recall 0.44762, and three false-positive edges among 9,573
negative pairs.  This is a more honest training condition, but a pairwise
classifier is still insufficient for the deployed global partition.

Artifacts:

- `/mnt/pool/sqy/results/RADIO-GS/output/v4_3_scannet_fragment_affinity_f05_calibrated_20260904`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_3_scannet_fragment_affinity_sam_like_object_equal_20260904`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_3_object_hypotheses_learned_group_fixed_assignment_20260904`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_3_object_hypothesis_ceiling_learned_group_fixed_assignment_20260904`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_3_object_hypothesis_text_learned_group_fixed_assignment_20260904`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_3_object_hypotheses_sam_like_affinity_fixed_assignment_20260904`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_3_object_hypothesis_ceiling_sam_like_affinity_fixed_assignment_20260904`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_3_object_hypothesis_text_sam_like_affinity_fixed_assignment_20260904`

## Same-view cannot-link diagnosis

The original global grouping treats every pair of proposals from one view as a
hard cannot-link.  This prevents part/whole hierarchy proposals for the same
physical object from sharing a hypothesis.  Two isolated structural ablations
tested whether that constraint explained the remaining fragmentation.

| Policy | hypothesis count (four scenes) | raw single | canonical single | canonical top-3 | generic consensus all-token |
|---|---|---:|---:|---:|---:|
| exact same-view cannot-link | 343/260/376/263 | 0.25623 | 0.22817 | 0.26484 | 0.09556 |
| only disjoint same-view pairs cannot-link | 210/163/217/158 | 0.23310 | 0.20926 | 0.23795 | 0.08282 |
| exact grouping, relaxed part/whole assignment | 343/260/376/263 | 0.23619 | 0.17960 | 0.20944 | not promoted |

Relaxing grouping improved generic-negative text-to-oracle-hypothesis R@1 from
about 0.258 to 0.345, because there were fewer duplicate candidates, but the
corresponding extent became impure and final mIoU fell.  This is direct evidence
that identity retrieval and extent association are currently misaligned.

This ablation also found a deterministic top-2 tie bug: three or more identical
nested proposals could evict a seed's own hypothesis by column order, leaving a
zero-responsibility hypothesis with no prototype.  Top-2 now reserves the seed's
own slot only when an exact tie would otherwise omit it.  The builder had
correctly rejected the bad artifact; no invalid waldo memory was written.

Artifacts:

- `/mnt/pool/sqy/results/RADIO-GS/output/v4_3_object_hypotheses_disjoint_same_view_cannot_link_20260904`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_3_object_hypothesis_ceiling_disjoint_same_view_cannot_link_20260904`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_3_object_hypothesis_text_disjoint_same_view_cannot_link_20260904`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_3_object_hypotheses_exact_group_disjoint_assignment_v2_20260904`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_3_object_hypothesis_ceiling_exact_group_disjoint_assignment_v2_20260904`

## All-view hierarchy supervision

The noise-matched cache was then extended from cross-view edge supervision to
all-view supervision.  Same-object overlapping whole/part proposals became
positive examples, while spatially disjoint same-view proposals remained hard
cannot-links at deployment.  This directly improved the held-out ScanNet
partition rather than only its independent edge score:

| Training scope | pooled AP | precision | recall | mean object fragmentation | object-best-hypothesis recall | merge impurity |
|---|---:|---:|---:|---:|---:|---:|
| cross-view only | 0.58476 | 0.96907 | 0.44762 | 3.4661 | 0.31460 | 0.00184 |
| all views with hierarchy supervision | 0.77347 | 0.98872 | 0.52183 | 2.3851 | 0.62123 | 0.00000 |

The all-view result was stable across the four held-out scenes: minimum
object-best-hypothesis recall was 0.58974 and every reported hypothesis was
pure under the synthetic oracle labels.  This validates the need to learn
same-view hierarchy relations.

LERF transfer nevertheless failed.  It retained 307/202/282/191 hypotheses;
the largest real-scene cluster contained 61 fragments, indicating a remaining
domain-shifted over-merge tail.  Raw single/top-3 ceiling was 0.23725/0.27501
and canonical single/top-3 was 0.19387/0.22866, below the strict baseline.
Frozen text-to-oracle-hypothesis R@1 rose to about 0.351 and R@3 to 0.688, but
generic consensus all-token mIoU was only 0.08730 and the best rank-union top-3
diagnostic was 0.14962.  Better retrieval over fewer candidates therefore did
not compensate for impure real-scene extent.

Artifacts:

- `/mnt/pool/sqy/results/RADIO-GS/output/v4_3_scannet_fragment_affinity_sam_like_object_equal_partition_metrics_20260904`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_3_scannet_fragment_affinity_sam_like_all_views_20260904`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_3_object_hypotheses_sam_like_all_views_affinity_20260904`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_3_object_hypothesis_ceiling_sam_like_all_views_affinity_20260904`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_3_object_hypothesis_text_sam_like_all_views_affinity_20260904`

## Static isolation repair

The full v4 regression exposed 21 static-isolation violations: nine formal
builders/evaluators still imported method-neutral helpers from the quarantined
development pipeline, and three existing comments contained prohibited method
terminology.  The shared data, camera, text-cache, polygon-mask, and metric
helpers now live in `radio_gs.v4.evaluation.lerf_common`; every formal path was
redirected to it.  The v4 static isolation audit now reports zero violations.
No historical method implementation was copied into the new path.

## Default restoration receipt

The current default was restored to full-hypothesis normalization followed by
top-2 retention and probabilistic fragment union.  Rebuilding all four scenes
reproduced the previous best groups and every numerical tensor exactly:

- group equality: true for all four scenes;
- maximum fragment-assignment error: 0;
- maximum null-probability error: 0;
- maximum observed-membership error: 0.

Restoration artifacts:

- `/mnt/pool/sqy/results/RADIO-GS/output/v4_3_object_hypotheses_restored_all_fragment_20260904`

## Decision

The surface carrier, valid region-level query keys, and canonical sum posterior
remain retained.  The current best LERF result is still only a development
diagnostic near 0.158, not a reasonable 0.4--0.5 complete result.  No formal
LERF3D number exists because no element-domain 3D target authority is present.

The next method step is not another threshold, top-k, carrier, or completion
normalization ablation.  The fragment cache now has a first deterministic
SAM-like noise model and object-equal sampling, but the deployed model still
scores edges independently and the agglomeration still makes order-sensitive
decisions.  The next association model must optimize a whole fragment set under
cannot-links/null and report fragmentation, merge impurity, object recall, and
held-out stability directly.  Completion must then consume that noisy fragment
set and predict spatial distribution and total mass separately.
