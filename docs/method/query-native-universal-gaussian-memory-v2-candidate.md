# Query-Native Universal Gaussian Memory v2 candidate

This document describes an experimental architecture, not the promoted final
method.  Universal Field v1 and its frozen benchmark rows remain authoritative
until the gates in ADR 0003 pass.

## Representation and query boundary

For Gaussian `i`, the deployment scene owns one latent and reliability vector:

\[
z_i\in\mathbb R^{512},\qquad r_i\in\mathbb R^5.
\]

A frozen pretrained encoder and a small modality adapter produce a
model-independent packet:

\[
Q_m=A_m(E_m(q_m)),\qquad m\in\{text,image,prompt\}.
\]

The shared decoder directly predicts the Gaussian Query Posterior:

\[
P_i(q)=\sigma D(z_i,r_i,Q_m,e_i^{prompt}).
\]

It does not decode `z_i` into a DINO, SAM or SigLIP visual descriptor.  Those
models may provide source-only response, ranking, correspondence, mask or
boundary supervision.

## Efficient identity--extent factorization

The dense reference implementation first computes an identity unary and then
an extent residual:

\[
s_i^{id}=\operatorname{LSE}_j\langle K(z_i),Q_j\rangle,
\]

\[
P_i(q)=\sigma(s_i^{id}+\Delta_i^{extent}).
\]

When a trustworthy pretrained scalar response already exists, it is replayed
as an identity prior.  The learned identity residual starts at exactly zero,
so extent training cannot silently destroy localization.  Registered prompt
seeds clamp strong positive/negative evidence; NaN denotes unknown.

The production implementation may precompute `K(z_i)` and execute extent only
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
