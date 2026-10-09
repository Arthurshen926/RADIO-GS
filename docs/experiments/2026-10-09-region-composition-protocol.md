# Existing-region composition capacity, diagnostic only

Uses the existing 6139 single-region candidate alpha arrays, same eight requests,
seven fields and old 120 sources. No changes to the running scope intervention.
Wait for its complete24 prediction barrier before opening any target labels here.

This is an explicitly LABEL-ASSISTED diagnostic, never an inference method:
nearest-resize target GT to192x256, threshold stored alpha>0.5, greedily add the
region yielding largest strictly improving IoU of the 2D mask union. Start empty,
ties choose lowest index, stop at no improvement; no tuned number of components.
Save the entire improvement trace and region IDs. This greedy process need not
find the global best union. Do not call it a true upper bound.

Then union the corresponding original Gaussian subsets B_positive[:,r]>0 and
member_available, and actually re-render that Gaussian union. Report BOTH the
2D union proxy and actual-subset alpha IoU/precision/recall at the same192x256.
They need not be equal: translucent component accumulation changes union alpha.
Report proxy foreground pixels missing in actual projection as a monotonicity
check without assuming the rasterizer guarantees mathematical set equivalence.

This answers whether a particular feasible multi-region combination outperforms
single-coordinate candidates. It does NOT establish that query-only inference
can identify that combination, certify capacity of other representations, improve
formal benchmark accuracy, or authorize deployment of GT-derived identities.
No diagnostic payload is loaded by standard inference or the scope intervention.
