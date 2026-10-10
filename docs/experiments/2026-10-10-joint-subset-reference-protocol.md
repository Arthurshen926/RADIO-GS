# Joint selected-silhouette reference under fixed partial evidence

Sequential development, not independent test or deployable method. Four LERF
scenes/68 texts, same72views/5137regions, exact cached teacher acceptances and
mass>0 eligibility. Positive=OR accepted existing partial labels;
negative=AND accepted existing partial labels; no accepted regions means unknown.
Source observed pixels and candidate region identities are unchanged. Do not
recompute raw-mask-union morphology or add detections/frames/target labels.

Three controls: source_parts (fixed source-normalized initializer),
joint_no_footprint, joint_footprint (PRIMARY). Same initializer and 240 steps,
Adam lr0.05, member temperature0.5->0.1, deterministic source-view cycle.
Geometry/field immutable, observed Gaussians trainable only as query-transient
membership logits. Actual member-opacity alpha, v10 cutoff10.5/255 transform,
balanced positive/negative BCE plus known-pixel overlap. Optional footprint
weight0.1, opacity-weighted unoccluded projected area summed over active views,
normalized by initial member cost; same cost/cameras in both solver controls.
All constants inherited as one reference solver package, no threshold/step sweep.
No-source-positive queries retain initializer. Source P/N readout and joint
objective are different measures: this tests the whole joint-solver function,
not an isolated causal estimate of a single loss or exact v10 parity.

All68 predictions/all arms before any scoring. Report actual-set IoU/response/AP,
P/R, per-scene changes, source objective, runtime, immutable geometry, no-default.
This explicitly USES source partial labels and cameras plus per-query fitting.
It is a CAPABILITY REFERENCE ONLY, not acceptable standard inference, learned
acceptance, field compression, global upper bound, or unified-five performance.
If it recovers, transfer that complete member function into query-independent
field state/shared inference using non-evaluation conditions. Do not store these
benchmark-query fitted answers as deployed field state or train against them.
