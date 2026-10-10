# Preserve native view confidence in the same compiled union decoder

Sequential development after some scope_constraint_readout_v3 scores were seen;
not an independent test. Same complete4LERF/68texts, exported fields, current query
vectors, semantic prior, accepted regions, threshold>.5, positive mass eligibility,
union composition and finite member formula. Only restore the EXISTING native
confidence rule from part_conditioning.prepare_parts: per-view max of positive
margin clamp(2*p-1,0,1), with zero-mass regions excluded. Multiply both per-view P
and N by this SAME confidence. Binary controls keep confidence1 as in v3. This is
not a confidence hyperparameter sweep or a claim of probabilistic calibration.

Four arms binary_partial/binary_scope(cached native controls),confidence_partial/
confidence_scope(PRIMARY). Do not use privileged teacher acceptance. Historical
mixed diagnostic payload is read solely for cached native controls and field-derived
semantic/query inputs; teacher member arrays are not inputs to new computations.
All new predictions for all scenes before scoring; all semantic scores unchanged.
No training, query optimization, source masks/cameras, GT in prediction or per-case
winner assembly. Conditional acceptance and actual membership are still unsolved.
No default promotion absent complete performance evidence. No whole-five claim.
