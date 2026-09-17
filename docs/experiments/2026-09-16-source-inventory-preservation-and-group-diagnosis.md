# Preserve source evidence before learning object extent

This iteration found and fixed an upstream evidence-loss defect. The component
source-root builder replaced **all training-view query-free regions** with
text-grounded connected regions. It retained only the old validation regions.
Consequently adding grounding supervision also removed source semantic and
structural evidence not represented by the limited grounding vocabulary.

The replacement was explicit in `prepare_component_source_root.py`:
`concat(grounded_inventory, old_regions[validation])`. A frozen native decoder
cannot preserve information that has already been removed from its source
inventory. This does not invalidate the earlier learned-space forgetting
measurements: upstream evidence loss and downstream learned-readout loss coexist.

## Repairs implemented

1. Preserve every original region, mask, view index, quality value and native
   descriptor in its original row. Append grounded regions; deduplicate only
   exact same-view mask additions. Nested part/whole masks remain distinct.
2. Use the same visibility rule for lifted region probabilities and `seen`.
   Previously a tiny contribution mass below the visibility threshold could be
   divided by itself to produce probability 1 on an element marked invisible.
   The compiler now emits zero region evidence for these elements, leaving them
   unknown. It does not assert an outside-object negative label.
3. Implement a conservative source-group association diagnostic. Associations
   require native semantic agreement, co-visible overlap, comparable support
   sizes, mutual-best partners per view, distinct views and complete-link
   consistency. The co-visible rule prevents occlusion from being treated as
   disagreement. This grouping experiment is not promoted into inference.

Existing sealed datasets/checkpoints were not overwritten. Rebuilt roots are
`/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916/{scene}`.
The original field remains untouched; source geometry and query encoders match.
No benchmark labels or target RGB enter source reconstruction or association.

## Actual integrity checks

| Scene | Original parent inventory retained | Rebuilt total regions | Unsupported nonzero memberships after repair |
|---|---:|---:|---:|
| figurines | 124 / 124 | 347 | 0 |
| ramen | 200 / 200 | 418 | 0 |

The previous component roots had 255 / 244 total regions, including validation.
The repaired roots retain 92 / 174 original training regions in addition to
223 / 218 grounded additions. Every original mask, view, quality and native
SigLIP descriptor was checked for exact equality. Both roots successfully
recompile all three query modalities and both source train/validation splits.

The old compiled roots contain 2,152 / 4,849 hard region memberships on elements
invisible from the corresponding view. Unsupported soft weights account for
1.01% / 0.98% of total region weight. This is a real consistency defect; its
magnitude does not justify attributing most retrieval failure to it.

Thirty relevant tests pass, including exact parent evidence preservation,
part/whole preservation, view-sensitive duplicate handling, low-mass visibility,
occlusion-aware association, semantic incompatibility and transitive-bridge
rejection, plus existing initialization and supervision behavior.

## Why group-first alone was insufficient

The diagnostic separately constructs source-only candidate masks, then evaluates:

- Best single candidate selected using GT, explicitly an oracle diagnosis.
- Top-1 fixed cosine using its original source-native descriptor.
- The same descriptor projected through the existing 64D native basis.
- A descriptor pooled from the unchanged field's native element directions.

Only the first selection accesses labels. All candidates and descriptors are
sealed before evaluation. These inventories are **diagnostic artifacts**, not
new deployment feature banks or a second persistent per-element semantic map.
The targets are category unions; a best-single-group ceiling is not the ceiling
for unions, new hierarchies or all possible grouping methods.

| Scene | Existing candidate oracle IoU | Native descriptor top-1 IoU | Field-mean top-1 IoU |
|---|---:|---:|---:|
| figurines | 40.52% | 25.99% | 22.07% |
| ramen | 34.35% | 10.19% | 8.70% |

Some ranges are missing independently of ranking: existing single-candidate
oracle IoU is only 13.0% for chopsticks, 5.1% for corn, 13.5% for kamaboko and
20.2% for wavy noodles. Conversely, plate and sake cup have candidate oracle IoU
64.1% / 58.5% but native top-1 only 3.9% / 0%, showing a separate association
problem. Neither is repaired by preserving an arbitrary MLP's input features.

Co-visible conservative merging adds 29 / 41 candidates. It changes oracle IoU
to 42.96% / 34.99% and native top-1 to 27.08% / 10.69%. This is too small and too
incomplete to justify replacing the current field readout. Full-world overlap
without visibility performed worse; all trial results remain recorded.

Restoring the query-free inventory gives candidate oracle IoU 41.04% / 36.26%.
The field-mean descriptors in this test still come from the old field, which
was built without restored evidence. Their results cannot be used as the
performance of a retrained field or proof that added evidence is harmful.

## Native fusion measurement and limits

Final comparisons use the **exact same element support** in both arms, frozen
geometry, the same 56 / 71 annotated text observations, verbatim native queries,
and a rendered `(cosine + 1) / 2` score without sigmoid calibration or threshold
search. Raw native teacher fusion, source-only SVD64 fusion, and the unchanged
old field are evaluated together. Values are recorded in the summary artifact
linked below. These AP values are not directly interchangeable with the earlier
sigmoid-rendered full-field evaluation protocol.

| Scene | Raw native fusion AP before → after | SVD64 fusion AP before → after |
|---|---:|---:|
| figurines | 56.66% → 58.35% | 56.02% → 57.68% |
| ramen | 22.15% → 31.67% | 22.54% → 30.87% |

Common support is exactly 71,739 / 168,801 elements. The unchanged field's AP
is exactly equal across arms (error 0). This rules out a changed evaluator or
valid-domain size as the explanation for the measured source-fusion gain.
It does not prove improvement after retraining and calibration.

The two source fixes are evaluated together; this trial does not numerically
assign separate causal shares to inventory restoration and visibility gating.
The unchanged old field is a control. Source-only support intersection avoids
comparing a larger valid carrier against a smaller one. An earlier preliminary
run used different valid supports (24 / 160 elements differed); use the final
`fusion_common` artifacts, not `fusion_matched`, for the comparison.

## Remaining work and promotion boundary

The repaired evidence and compiled episodes are ready. The old grounded-text
cache, conflict audits, source-fusion initialization and trained model depend on
the old inventory/hash and must not be reused as if they described the new one.
Rebuild their bindings and targets, fit the native anchor from preserved evidence,
then freeze that anchor before structural training. Evaluate image and prompt
queries as well as text after reconstructing the field.

This iteration does not claim a new final three-query model, solved part/whole
hierarchy, independent final-test improvement, or completed NVOS/ScanNet-OVS
validation. The larger candidate space can also change mutual-best cross-view
matches; complete episode compilation is not proof of correspondence accuracy.

## Artifacts

- `paper/artifacts/preserved_source_inventory_integrity_20260916.json`
- `paper/artifacts/source_inventory_repair_summary_20260916.json`
- `paper/artifacts/{scene}_{component_retrieval_pipeline_20260915,preserved_source_inventory_20260916}_fusion_common_20260916.json`
- `paper/artifacts/{scene}_{source,merged,visible_merged,preserved}_group_diagnostic_20260916.json`
- Corresponding `.png` files compare GT, oracle range, field-mean top-1 and
  original native top-1 on the first annotated view.
