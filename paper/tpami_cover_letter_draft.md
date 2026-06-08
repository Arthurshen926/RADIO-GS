# TPAMI Cover Letter Draft

Dear Editor-in-Chief and Associate Editors,

We are pleased to submit our manuscript, **"CTF-GS: Compact Teacher Feature
Fields for Open-Vocabulary 3D Gaussian Scene Understanding"**, for consideration
in *IEEE Transactions on Pattern Analysis and Machine Intelligence*.

This work studies a compact foundation-feature representation for 3D Gaussian
scenes. Instead of storing raw high-dimensional teacher features per Gaussian or
training a scene-specific classifier, CTF-GS reconstructs frozen RADIO features
from a Hybrid Gaussian Code Field and exposes the same compact scene memory
through three readouts: rendered-view open-vocabulary localization, direct
Gaussian primitive selection, and direct point-query transfer.

The submission makes the following contributions:

1. A compact teacher-feature field for 3D Gaussian scenes that supports both
   rendered 2D feature maps and direct 3D primitive/point queries.
2. A unified multi-head architecture with compact-to-teacher reconstruction,
   view-space feature alignment, feature-quality/visibility heads, and
   frozen-head adaptor consistency.
3. A View-to-Primitive Registration training bridge that transfers registered
   multiview evidence into the compact field while avoiding a VPR feature cache
   at inference for the main direct-3D readout.
4. A protocol-separated evaluation on LERF-OVS rendered-view grounding,
   OpenGaussian-style direct 3D object selection, VALA/OpenGaFF-8 ScanNet direct
   point queries, storage/efficiency analysis, teacher-vs-student probes, and
   failure analysis.

The manuscript is intended for TPAMI because it addresses a central problem in
computer vision and pattern analysis: how to turn dense 2D foundation-model
features into a compact, reusable 3D scene representation with auditable
open-vocabulary behavior across multiple query protocols. The work extends
recent 3D Gaussian open-vocabulary understanding beyond a single rendered-view
metric by explicitly connecting rendered feature maps, primitive-level object
selection, and point-level scene querying.

All experiments are organized with fixed provenance artifacts. The manuscript
distinguishes locally evaluated rows from source-anchored external context rows,
and the supplementary material provides protocol controls, additional
qualitative results, and reproducibility instructions.

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
