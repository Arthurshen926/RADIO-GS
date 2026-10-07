# Fixed shared-support training study

Starting revision b831d03. Keep the accepted physical member energy, frozen
native channels, geometry, source masks, exclusion eligibility, candidate policy,
default physical scale and all readout thresholds. Only training support forms
change. No extra labels, query-time source access or benchmark-specific decoder.

Current fitting presents only one positive dominant Gaussian. Deployed point
inputs contain uncertain ray mixtures; scribbles contain multiple rays and may
include negative evidence. The existing signed member product is untrained as
a conditional source episode. This establishes a distribution mismatch, not
proof that it uniquely causes observed false additions.

Two fixed arms, both 2000 updates from the identical initialization and exactly
the original seeded source-region/first-pixel sequence:

* mixed_unsigned cycles anchor, single ray, four positive rays, four positive rays.
* mixed_signed cycles anchor, single ray, four positive rays, four positive plus
  four negative rays.

Additional pixels are distinct eroded source interiors/exclusions with observed
dominant contributors. Keep all weighted contributors in each selected ray.
Use the deployed deterministic raster adapter, then the identical observed-support
filter and signed member product. Negative supports use only the original known
comparable-scale disjoint teacher masks, never unknown pixels. A missing reliable
negative retains unsigned pooling and is counted. No hard 2D or training clamp.
The separate support RNG draws matching positive/negative pixels in both arms;
the original episode RNG is untouched. One rendered loss and optimizer step per
update; the signed arm performs an additional member evaluation in 1/4 of steps.
Report that training compute difference; no runtime/storage increase is intended.

Run both arms on figurines/ramen (same 18 feature views) and fern (same 12 views).
Save all four LERF text predictions and both signs of both fern fits before opening
any target labels. Reuse native LERF query vectors; native response must remain
bit-exact. Score the retained response and actual selected subset separately.
Evaluate every default image/2D/3D teacher region, including unsupported zeros.

A candidate must improve actual LERF subset IoU on both scenes; no scene's image,
2D or 3D default mean may regress; fern's positive and signed response IoU, AP
and actual subset IoU must not regress relative to the common physical fit.
If an arm passes, confirm unchanged on teatime and waldo before promoting across
the full LERF cohort. If both pass, choose greater two-scene actual text mean;
tie prefers unsigned for lower training compute. Otherwise keep the current
candidate and all failures. No support-count, mixture-weight or threshold sweep.
This is historically opened development, not independent final evidence.

Source-only reproducibility amendment, before any new public predictions/scores:
the single-support refactor replay has identical initialization and logged sample
order but small early loss differences and nonidentical final learned tensors.
Repeat the exact b831d03 field/fitter twice on fern with identical settings and
compare to the saved original and refactor replay. These are numerical controls,
not competitive method arms or a seed/parameter selection. Keep the predeclared
method gates and report training repeatability separately from deterministic
deployed-query replay; do not attribute small differences to architecture alone.
