# Follow-up analytic scope bound

The completed first diagnostics show opposite error directions: LERF actual sets
overselect, while NVOS flower/orchids strongly underselect with negligible change
when negative evidence is removed. All supported UQIS 3D anchors belong to their
requested targets, yet member extent remains poor.

Use the exact normalized-prototype physical energy, without changing any field:
`logit_i <= tau * (1-bias) - ||x_i-centre||²/(2*scale²)` in physical units,
because normalized group cosine is at most one. Thus the current 0.5 element
cutoff permits members only inside radius
`scale * sqrt(max(0, 2*tau*(1-bias)))`, except any explicitly clamped anchor.
Verify the bound against saved scores rather than assuming it holds numerically.

For all supported UQIS queries, partition target elements into unobserved,
observed but outside this envelope, observed and inside but missed, and selected.
Record actual target spatial scale only as a diagnostic ratio; do not use it for
inference. Also report observed-filtered 2D support association separately from
the previous raw support mass. Preserve unsupported cases in full-cohort results.

For all eight NVOS cases, project this optimistic member bound through the same
retained response renderer and report an optimistic target-recall bound. It is
not a mask IoU upper bound, an achievable decoder, or an improved method result.
The bound keeps the fitted temperature, bias and scene-default scale fixed, so
it does not bound a retrained architecture. No oracle scale/anchor or threshold
search, no training, no altered predictions.
