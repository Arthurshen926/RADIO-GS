# Fixed interpretation checks for element relation construction

Specified while the full construction cohort is running, before any new
benchmark prediction or score is available. These checks never fit weights,
select checkpoints or change the two-arm protocol.

1. Full-UQIS post-score observation-contribution diagnosis: retain all67 3D
   inputs and the known-correct supported subset. Repeat the existing observed
   target recall strata for both new arms and the protected rendered control.
   Report actual precision/IoU as well as recall; high-minus-low exposure gaps
   pair strata within the same query and resample physical scenes. Contribution
   remains an exposure proxy, not a measurement of rendered-loss gradients.
2. All30 conditional exports, held-out source relation audit: replay the fixed
   development pairs and Gaussian samples (seeds20261011/20261012). Among unique
   sampled rows, permute paired semantic/local cosine features using seed20261013.
   Separately permute residual geometry features or element exposure confidence,
   using the same permutation. Preserve the group feature and analytic base
   logit. Also remove the residual while retaining the jointly learned groups.
   Report BCE and its differences from original, and correction magnitude and
   within-query variation. This tests sensitivity to feature binding in source
   queries; it is not proof of independent benchmark utility, an on-manifold
   alternative method, or an isolated effect of the original training factors.

All source counters use partial SAM-derived relationships, not complete-object
ground truth. A residual removal may expose co-adaptation with grouping and
must not be described as the independently trained element-only arm. Never
substitute any permuted/removed result into deployed predictions.
