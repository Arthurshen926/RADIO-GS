# Integrated native-evidence Gaussian query program

Predeclared during relation training, before any new cohort scores. The common
candidate always uses the `cross_view_anchor` field on every benchmark/scene.
No checkpoint is selected according to forthcoming task results.

Shared implementation: `radio_gs.query_field.query_program`. Pure image instance
queries first fuse SigLIP and DINO maximum-patch ranks across all eligible Gaussian
anchors, then apply the original spatial NMS/count. Compute the unchanged member
scope for each candidate, match every selected Gaussian local vector against
every current image patch in both directions, and fuse this correspondence score
with the candidate's native semantic rank. Select one coherent member hypothesis.
Empty scopes remain empty, inferred anchors are not forced positive. Registered
supports, text queries and explicit union conditions retain the common original
member semantics; signed exclusions remain in the common kernel.

Category partitions consume complete-menu native unaries and member supports via
the separately frozen bounded category-evidence reader. All scene state remains
the exported Gaussian field. The program is shared code, not a source memory,
new teacher, extra label-trained head or per-benchmark scene map.

The combined image rule was not evaluated by the previous four-arm identity study;
its interaction with newly trained scopes is a new hypothesis. Current fused
Gaussian local vectors still do not reproduce v10's original patch distribution.
This is not yet functional native-field reconstruction or a complete uncertainty
model. These limitations must remain in the results and future work.

Run all67 image queries with this rule. Reuse the already computed predictions
for the exact same cross-view fields on other inputs, proving those paths are
unchanged and binding all hashes. This is computation reuse under one declared
program, not per-query best routing. Complete the new predictions before scoring.
Assemble all268 UQIS inputs and the full five-task table, using the predeclared
typed category output; retain the separate training/readout ablations. Verify
standalone field replay and exact nonimage equivalence. Report all regressions.
