# Source-weighted two-mode native semantic field: bounded pilot

The two registered source-only pilots are complete and both fail admission.
No benchmark evaluation, dependent-head fitting, threshold sweep, or mainline
promotion follows this result. This rejects the particular two-mode estimator
and readout below; it does not disprove every multimodal representation.

## Method and input contract

The hypothesis is that collapsing incompatible source-region descriptors into
one mean destroys distinctions that no later query readout can recover. The
candidate retains two query-independent semantic modes per element in the same
original64-dimensional native basis. Source votes use the existing renderer
contributions and overlap partition, with exactly the same geometry, training
views, native descriptors and observed support as the baseline.

The fixed spherical weighted2-means initialization uses the weighted mean and
the source vote maximizing `weight*(1-cosine_to_mean)`, followed by8 updates.
Assignments use angular distance. Each centroid retains the weighted mean of
raw projected coordinates; mode priors are actual source contribution masses.
Thus the prior-weighted centroid mean reproduces the original collapsed fusion.
No heldout view supplies a fitting vote.

The query score is `tau*log(sum(prior*exp(native_component_cosine/tau)))`.
Unlike unweighted max pooling, a weak mode pays its source-mass prior penalty.
The query uses its full original native norm, including dimensions outside the
64-dimensional projection. Scores are computed on elements first, then passed
through the unchanged observed-normalized renderer and region averaging.
Rendering mode vectors before the nonlinear readout would be a different method
and is not used.

One global temperature per scene equals the weighted mean training angular
residual `1-cosine(vote, assigned_center)`, floored at1e-4 for numerical safety.
This is an energy-scale heuristic, not a calibrated posterior or an optimal
temperature claim. No query-specific or heldout-fitted temperature is used.

The element vector grows from96 to160 float32 coordinates: two64-dimensional
semantic blocks plus the copied32-dimensional structural suffix. One additional
float32 mixture-prior scalar is required. Persistent element storage therefore
grows **384→644 bytes per element (+67.71%)**, excluding unchanged geometry,
support and shared decoder basis. The pilots use176,974,420 versus105,525,120
element bytes for figurines, and52,401,636 versus31,245,696 for ScanNet0000.
The copied structural suffix is not revalidated and old calibrated probability,
scope and prompt heads are not reused. This is an increased-capacity native
readout control, not equal-capacity evidence for architecture superiority.

Registration: `output/native_mixture_next31/registration.json`.
Implementation: `radio_gs/training/native_semantic_mixture.py`.

## Admission and complete pilot results

Both source pilots must independently improve heldout regional text **and**
image response MSE by at least1%, without increasing centered MSE or lowering
teacher Top-1 agreement by more than1 percentage point. Both modalities also
need nonincreasing heldout boundary response MSE and one-sided over-response
error. Target benchmark queries, target RGB, masks and labels are not read.

Image queries are512 fixed evenly spaced training descriptors. Text queries
are the existing fixed source teacher text bank. Validation uses the original
two reserved source views per scene. The boundary gate uses source teacher
mask boundaries and `max(prediction-teacher_response,0)^2`, not labeled object
false-positive rate.

| Pilot / metric | Collapsed native mean | Two-mode mixture |
|---|---:|---:|
| figurines text MSE |0.000439451|0.000450695 (+2.56%)|
| figurines image MSE |0.010231940|0.010108955 (−1.20%)|
| figurines text teacher Top-1 |34.72%|36.81%|
| figurines image teacher Top-1 |11.81%|13.89%|
| ScanNet0000 text MSE |0.000342437|0.000342948 (+0.15%)|
| ScanNet0000 image MSE |0.005586500|0.005149654 (−7.82%)|
| ScanNet0000 text teacher Top-1 |41.72%|43.05%|
| ScanNet0000 image teacher Top-1 |21.19%|18.54%|

Text boundary MSE and text boundary over-response increase on both pilots.
Image centered MSE also increases on both, despite lower image total MSE.
Therefore the image MSE gain cannot be described as uniformly better query
discrimination: part of it changes response offsets, and ScanNet image ranking
agreement actually falls. Both pilots are rejected under the preregistered
rule. The text Top-1 improvements remain diagnostic observations, not final
retrieval gains.

Estimated temperatures are0.0442696 and0.0258277; neither reaches the numerical
floor. Of observed elements,43.97% and51.34% have both mode priors above0.1 and
intermode cosine below0.9. That definition is a reported diagnostic only, not
an inference or admission threshold. Distinct modes are present; the negative
result is not simply a degenerate duplicate-component implementation.

The old collapsed source mean reproduces within1.55e-6/8.94e-7. The two-mode
weighted centroid reconstruction reproduces that same mean within8.94e-7/
5.96e-7. Basis and support remain unchanged. Six CPU tests cover component
permutation, identical-component baseline equivalence, full native query norm,
source-mass priors, empty/single modes, and invalid evidence.

## Interpretation and evaluation limits

Two appearance/semantic clusters do not by themselves establish whether their
votes describe the same physical object, different objects, or a whole and a
part. The fixed weighted log-sum-exp can improve some query ranks while also
raising unwanted responses or damaging other query relations. More modes or
more aggressive max-like temperatures are not justified by these results.

The boundary pseudo-target itself normalizes the mean of overlapping regional
native descriptors. It already collapses ambiguous evidence, so that gate may
penalize a useful multimodal response merely for differing from that mean.
Neither it nor source region descriptors are verified physical object truth.
We retain the registered rejection, disclose this limitation, and do not revise
the gate after observing results. Independent scope/identity evidence is needed
before assigning positive and negative meaning to those conflicting modes.

Full reports and opt-in160-coordinate artifacts:
`output/native_mixture_next31/lerf_figurines/` and
`output/native_mixture_next31/scannet_scene0000_00/`.
Final binding: `paper/artifacts/native_mixture_source_pilots_next31.json`.
