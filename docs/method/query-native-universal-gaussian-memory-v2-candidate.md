# Query-Native Universal Gaussian Memory v2 candidate

This document describes an experimental architecture, not the promoted final
method.  Universal Field v1 and its frozen benchmark rows remain authoritative
until the gates in ADR 0003 pass.

## Representation and query boundary

For Gaussian `i`, the deployment scene owns one latent and reliability vector.
The query boundary reads the post-fusion coefficient by default:

\[
c_i=\mathrm{Fusion}(z_i,\mathrm{spatial},r_i)\in\mathbb R^{512},
\qquad r_i\in\mathbb R^5.
\]

A frozen pretrained encoder and a small modality adapter produce a
model-independent packet:

\[
Q_m=A_m(E_m(q_m)),\qquad m\in\{text,image,prompt\}.
\]

The shared decoder directly predicts the Gaussian Query Posterior:

\[
P_i(q)=\sigma D(c_i,r_i,Q_m,e_i^{prompt},G_i).
\]

It does not decode `z_i` into a DINO, SAM or SigLIP visual descriptor.  Those
models may provide source-only response, ranking, correspondence, mask or
boundary supervision.

## Efficient identity--extent factorization

Reading raw `local_codes` is an explicit ablation because their gauge is
internal to a scene. `CanonicalGaussianField.query_memory` exposes three
controlled arms: raw local code, fused coefficient, and decoded RADIO under
one globally fixed low-dimensional projection. A constant-size low-rank scene
FiLM may be used for gauge alignment; a Gaussian-indexed adapter is forbidden.
The implemented shared-model candidate uses a zero-initialized rank-8 scene
FiLM.  Three development seeds improve heldout scene-query transfer over the
uncanonicalized coefficient control, but all still fail one strict class-set
gate.  It therefore remains an ablation component, not a promoted dependency.

Separating pair-local identity from centered query-set competition closes the
heldout source gate (`21/21`), validating the architectural decomposition.
Its paper8 row remains below the retained compact decoder on all three splits,
so decision-preserving source supervision is still required before promotion.

The image-query branch now has a real physical-object training contract:
cross-view DINO/RANSAC/cycle-confirmed pairs provide query and target views,
continuous target-view exact-MPR membership provides extent, explicit
different proposals provide negatives, and all other support is unknown.
Two of three independent source scenes pass with large/positive gains; the
sparse Waldo scene requires shared cross-scene training before benchmark use.

The categorical branch now owns label-free opacity-volume weights and a
class-balanced top-class/soft-IoU objective.  Its residual remains bounded by
baseline margin and predicted margin gain.  Because the current eligibility
model still harms some heldout queries, these additions are implemented but
unpromoted.

The dense reference implementation first computes an identity unary with a
learned temperature `tau` constrained to `[0.02,0.2]`, then an extent residual:

\[
s_i^{id}=\operatorname{LSE}_j\langle K(c_i),Q_j\rangle/\tau,
\]

\[
P_i(q)=\sigma(s_i^{id}+\Delta_i^{extent}).
\]

The highest-scoring 4--8 identity rows form separate anchors. Finite extent
features include relative xyz, Euclidean and scale-normalized distance,
relative Gaussian scale, opacity, anchor confidence and feature relation.
Extent is local evidence propagation, not unrestricted scene diffusion, and
cannot reduce the strongest identity-anchor logits.

When a trustworthy pretrained scalar response already exists, it is replayed
as an identity prior.  The learned identity residual starts at exactly zero,
so extent training cannot silently destroy localization.  Registered prompt
seeds clamp strong positive/negative evidence; NaN denotes unknown.

The production implementation may precompute `K(c_i)` and execute extent only
on retrieved anchors and their neighborhood.  The first sentinel uses a dense
reference to establish numerical behavior before sparse optimization.

## Categorical query sets

For ScanNet, one shared function acts on each `(Gaussian, text query)` pair.
Query competition is represented only through permutation-invariant set
statistics.  There is no `(19,15,10)`-indexed output parameter:

\[
\hat s_{iq}=s_{iq}^{base}+
g_\theta(z_i,r_i,t_q,s_{iq}^{base},\operatorname{Pool}(\mathcal Q)).
\]

Reordering queries exactly reorders scores, and changing query cardinality
requires no architectural change.  A source-only query-set gate may abstain
and replay baseline scores, but its threshold must be global or source-gated;
benchmark labels and masks cannot select it.

The current ScanNet pilot still receives `s_base` computed by the retained v1
identity path.  It is a posterior-level bridge experiment, not the final
feature-decode-free implementation.  The final candidate must learn
`s_base = I(z_i,t_q)` across scenes and pass scene-query holdout before removing
the v1 identity path.

## Training order

1. Freeze L512 and train query adapters/direct posterior decoders.
2. Add source-row and source-view heldout gates.
3. Add image and prompt adapters.
4. Add cross-modal consistency only for independently trusted object pairs.
5. Open a low-rank latent residual only if decoder-only capacity is the proven
   bottleneck.
6. Compare RADIO regularization, RADIO initialization only, and no-RADIO under
   matched capacity.

NVOS's frozen RGB-assisted SAM3 compiler remains the public development row;
the query-native prompt path is initially a field-only fallback and UQIS
track, not a replacement selected for architectural neatness.

## Object-supervision contract

Object membership is three-state. Positive rows come from a trusted matched
mask; negatives come only from disjoint, semantically dissimilar competing
proposals; every other visible row is unknown and contributes no binary loss.
Source train views, validation views and final heldout views are disjoint.
Checkpoint selection and threshold calibration use validation only. An
unmatched heldout proposal is merely a proposal-generalization sentinel and
must not be described as cross-view physical-object supervision. Promotion
requires a separately gated same-object query-view to target-view episode.
