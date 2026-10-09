# Preserve menu-conditioned semantic functions on the Gaussian field

Extends ADR0014/16 within the user-authorized functional transfer direction.
Status: implementation and fixed full-ScanNet mechanism study, not promotion.

A category partition has a complete menu; it is not repeated single-instance
selection. Averaging descriptors before softmax does not preserve the weighted
mixture of per-mode category probabilities. Query-dependent entropy weighting
also cannot be reconstructed from one mean descriptor. Preserve normalized native
semantic atoms and a Gaussian-to-atom sparse measure inside the field export.
No class names, previous answers, source raster/view layout, masks or cameras
belong to this additional state. No runtime source loader or old task solver.

The generic category_partition entry uses the declared complete menu on every
atom, integrates probabilities with optionally retained entropy information,
then applies the existing bounded common-member corroboration. Other input
interfaces and their membership function remain unchanged. This is an explicit
typed semantic readout, not a new independent task-specific scene map. It does
not yet unify the semantic atoms with image-mode storage; storage sharing is
future work after functional validation, not a claimed existing innovation.

Three fixed readout controls use identical scene state and legal source views:
softmax of averaged logits, averaged per-atom softmax, and entropy-weighted
averaged per-atom softmax. Each also exposes prior-only output as a diagnostic
against the same shared-member corroboration. Temperature100 is inherited, not
selected on these scores. Zero-information support retains the unweighted
mixture; missing support retains the prior native fallback. All masks of carrier
availability are declared; additional source coverage is separate from operator
order. The reference is uncompressed; data and bytes must be disclosed.

v10 additionally uses runtime text-conditioned SAM evidence and a local auxiliary
channel. These functions are not transferred by this semantic primitive. No
full-v10 parity claim follows, nor a full-five-task claim from ScanNet alone.
