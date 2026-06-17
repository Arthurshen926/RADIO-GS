# TPAMI Cover Letter Draft

Dear Editor-in-Chief and Associate Editors,

We are pleased to submit our manuscript, **"CTF-GS: Compact Foundation-Feature
Gaussian Memory for Open-Vocabulary 3D Scene Understanding"**, for consideration
in *IEEE Transactions on Pattern Analysis and Machine Intelligence*.

This work studies a compact reconstructive foundation-feature memory for 3D
Gaussian scenes. Instead of storing raw high-dimensional RADIO features per
Gaussian or training a scene-specific classifier, CTF-GS stores low-dimensional
Gaussian latent codes with spatial context and reliability cues, reconstructs
RADIO-compatible scene features on demand, and uses the same compact memory for
rendered-view open-vocabulary localization, direct Gaussian primitive selection,
and direct point-query transfer.

The submission makes the following contributions:

1. A compact reconstructive RADIO Gaussian feature field that replaces explicit
   high-dimensional feature storage with latent scene memory, spatial context,
   reliability modeling, and global feature decoding.
2. A training framework that combines dense rendered RADIO reconstruction with
   sparse Multiview Primitive Registration and GT-free support calibration,
   enabling rendered 2D query, direct 3D primitive query, and ScanNet point query
   from the same stored memory.
3. A protocol-separated evaluation on LERF-OVS 2D and 3D open-vocabulary query,
   VALA-aligned ScanNet direct point query, storage/efficiency analysis,
   frame-wise RADIO comparisons, qualitative analysis, and component ablations.

The manuscript is intended for TPAMI because it addresses a central problem in
computer vision and pattern analysis: how to turn dense 2D foundation-model
features into a compact, reusable 3D scene representation with auditable
open-vocabulary behavior across multiple query protocols. The work extends
recent 3D Gaussian open-vocabulary understanding beyond a single rendered-view
metric by explicitly connecting rendered feature maps, primitive-level object
selection, and point-level scene querying.

All experiments are organized with fixed provenance artifacts. The main LERF and
ScanNet quantitative tables use the reproduced protocols reported in the
manuscript, while historical provenance notes and additional controls are kept
in the supplementary material for auditability and reproducibility.

This manuscript has not been published or submitted elsewhere. All authors have
approved the submission. Any use of AI-assisted tools, code repositories,
datasets, pretrained models, and third-party assets should be disclosed in the
final acknowledgements and submission forms according to IEEE policy.

Sincerely,

`<Corresponding author name>`  
`<Affiliation>`  
`<Email>`  
`<ORCID, optional>`

## Items To Fill Before Upload

- Full author list, affiliations, emails, ORCID identifiers, and corresponding
  author.
- Submission mode decision: upload the anonymous review PDFs unless the TPAMI
  portal explicitly requests author-visible manuscripts.
- Conflict-of-interest and funding statements.
- Dataset, pretrained-weight, and code-license disclosures.
- Any IEEE-required AI-assisted writing/tool disclosure.
- Prior conference/preprint relationship, if any.
- Suggested reviewers and excluded reviewers, if the submission portal asks.
