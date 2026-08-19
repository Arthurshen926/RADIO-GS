# Universal Field v1 with typed instance-aware readouts

## Method boundary

Every Gaussian stores one query-independent D512/L512 Canonical Capability
Feature and five reliability scalars.  Official DINO, SAM and SigLIP modules
are frozen capability operators; their outputs are supervision or
query-workspace tensors, not additional persistent scene fields.

The field and readout solve different problems:

1. the field reconstructs reusable appearance, semantic and structural
   evidence;
2. a typed posterior interprets that evidence under the legal input and output
   contract of a query family.

Primitive Readout-v0 remains the causal baseline.  The following typed
readouts are the promoted method.

## Text identity and object extent

For text query $q$, the field supplies the identity unary

\[
u_i(q)=\cos(e_i,t_q),
\]

where $e_i$ is derived from the same Canonical Capability Feature and $t_q$ is
the frozen text embedding.  Query-independent source-view official-SAM
proposals are associated across views and lifted to Gaussian membership
$r_{ik}$ with exact marginal responsibilities.  Proposal identity is scored by
the official SigLIP masked-crop representation.  The extent posterior is

\[
p_i^{\mathrm{ext}}(q)=\sum_k p(k\mid q)r_{ik},
\]

subject to a field-peak anchoring constraint: a region may complete the object
containing the identity peak, but may not replace that peak.  Localization
uses $u_i$ and segmentation uses $p_i^{\mathrm{ext}}$; they are rendered and
resized independently.  This factorization prevents broad regions from
diluting sparse text identity.

The promoted full-four development result improves LERF2D sample-micro mIoU
from `0.31417` to `0.39584`, preserves localization accuracy at `0.87981`, and
improves LERF3D mIoU from `0.33450` to `0.39684`.  Every scene improves.

## Mutually exclusive categorical posterior

For ScanNet OVS, class identity remains the normalized categorical field
posterior.  Let $m_i$ be the top-two class margin and $L_{\mathrm{sam}}$ a local
query-independent SAM-affinity operator.  The accepted correction is

\[
\tilde p_i = p_i + \alpha\,\mathbf 1[m_i<\tau]
             \left(L_{\mathrm{sam}}p-p\right)_i.
\]

Thus confident rows are immutable and uncertain rows receive only bounded
topological evidence.  Fixed parameters are 8 neighbors, 0.1 m radius,
SAM cosine threshold 0.5, $\alpha=0.25$, and margin threshold 0.03.  On paper8,
19/15/10-class mIoU changes from `0.33535/0.33370/0.42213` to
`0.33810/0.33617/0.42548`; all corresponding mAcc values also improve.
Unrestricted SAM categorical propagation is explicitly excluded.

## Registered prompt posterior

NVOS and Available-Nine SPIn-NeRF provide registered prompts and permit the
contract-specific transient RGB workspace.  The field first seals positive and
negative support.  Frozen official SAM then decodes box and signed-point
hypotheses independently.  Final consensus ranks masks by positive inclusion
and negative exclusion; field overlap and SAM confidence are tie-breakers.

Box and points are not concatenated inside the SAM prompt encoder.  The joint
call reduced NVOS macro IoU from `0.81776` to `0.68259`, showing that prompt
encoder interactions are not an additive posterior.  The promoted independent
selector reaches `0.81776` versus the field-only `0.52687`, without using target
masks for prediction.

SPIn uses the same registered-prompt interface after all nine Method-v1 fields
pass their field gates.  A historical carrier score is not a Method-v1 result.

## Promotion rule

A typed readout is promoted only when its frozen cohort gate passes and no
scene exceeds the allowed regression.  Development improvements are recorded
as development evidence; they are not automatically prospectively blind or
SOTA claims.  Failed candidates remain negative controls and cannot be selected
per scene.
