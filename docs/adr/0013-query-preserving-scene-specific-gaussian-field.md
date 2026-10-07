# Query-preserving scene-specific Gaussian field

Status: accepted research direction by the user on 2026-10-07; no accuracy promotion.

The active research branch is `codex/query-preserving-gaussian-field`, in the
independent worktree `/root/RADIO-GS-query-field`. Commit
`0f40545c2b93fa390e37038af8315c24b5bc57b5` preserves the complete nonignored v10
working state, including previously uncommitted files. The original worktree and
its index were not changed. External data and ignored outputs remain separately
versioned assets; the source snapshot is not a claim of a fully portable baseline.

## Decision and conflicts with earlier decisions

The target is one scene-specific Gaussian capability field, with native SigLIP
semantic coordinates, native DINO local appearance, and SAM-supervised
scale-conditioned grouping. These are owned subspaces on the same Gaussian
rows, not benchmark-specific maps. A native decoder and its query rule must be
paired, including bias and decoded norms. Initially retain sufficient native
capacity; measure compression separately after the member mechanism works.

This explicitly replaces the RADIO-required reconstruction contract of ADR0002
and older Canonical Capability Feature implementations for the new candidate.
The project name does not impose RADIO as an intermediate teacher. ADR0008's
frozen geometry, ADR0009's protected writes and ADR0011's distinction between
semantic response and actual subset rendering remain in force.

ADR0012 remains the description of baseline A (v10). Sharing observation utility
functions is no longer sufficient for the new candidate's unification gate:
text, image and registered prompts must consume a common member operator.
Category competition and the final rendering operator can remain typed. The
first category reader is a native semantic control; affinity-aware categorical
benefit is pending and must not be claimed from this interface.

## Data and deployment boundaries

Construction uses only manifest-authorized mapping RGB/cameras and native
DINO/SAM/SigLIP. Additional 3D instance/NR3D labels are a separate condition,
not an implicit part of this first implementation. A scene-specific fit never
authorizes use of evaluation labels or final query answers.

The standard scene package contains geometry, capability codes, grouping
parameters and any required affine codecs. Runtime accepts this package and
the current authorized input. Source photos, masks and tracking are construction
artifacts, not implicit query resources. Input encoders are declared shared
dependencies and their storage and latency are additional to per-scene cost.
Explicitly permitted source-assisted NVOS and other extensions must enter the
same membership core and have separate quality/cost reports. Existing v10
source-assisted tasks are preserved as comparators, not automatically approved
new standard-mode dependencies.

Estimated identities and 2D ray correspondences are not certain positive
Gaussian labels. Only explicitly registered positive elements can be clamped.
Alternative identity/scale evidence should be retained before selection.
The first implementation exposes alternatives and returns a coherent MAP
hypothesis; relational text binding and calibrated identity uncertainty are
still pending. The first image rule retains query patches but does not claim
to reproduce v10's full bidirectional source-patch matching.

## First bounded experiment

`configs/query_field/figurines_pilot.json` fixes six feature-training and two
development mapping frames, a 96x128 projection and 200 grouping updates before
new results are read. Geometry is the actual v10 Gaussian carrier. Geometry
was fitted on a larger mapping set, so this is a feature-view holdout, not
unseen-view geometry evaluation or an independent final test.

Recompile source assignments on this carrier. Do not reuse earlier 274805-row
memberships on the current 554467-row geometry. Cache receipts bind source
image, native region cache, SAM checkpoint/cache, DINO checkpoint, geometry,
camera report and configuration. Development features never initialize the map.

The control keeps the same native semantic/local channels with an untrained
group projection. The candidate fits only group codes and a scale gate with
actual Gaussian-rendered partial mask losses. Disjoint, comparable-scale
source masks can supply exclusions; nested masks and unobserved areas remain
unknown. These are teacher grouping targets, not verified whole-instance truth.

## Progression

1. Freeze A and its dependencies; record actual saved metric contracts.
2. Verify the Gaussian compiler, exact query algebra, and protected native field.
3. Establish conditional grouping/member competence on fixed development views.
4. Align the four query adapters and identity/scale inference; compare high-capacity B.
5. Compress under the same inputs, supervision and readout; validate isolated loading.
6. Expand across all five tasks and lock independent final evaluation.

Each transition needs measured evidence. A failed bounded pilot remains a
diagnostic, not a promoted replacement or a reason for unrestricted parameter
search. Proposed one-percentage-point quality and fourfold semantic-increment
compression goals are planning values until the comparable B baseline is fixed.
Performance retention must also be measured against A; an A-to-B contract loss
cannot be hidden by good B-to-compact retention. No top-conference novelty or
accuracy claim follows from the package structure alone.

## First implementation evidence, 2026-10-07

The fixed 200-step pilot and one 2,000-step optimization-budget control are
complete. A third, matched 2,000-step control transfers the established
next67/68 distinction into the shared member decoder: fit a relaxed selected
subset's alpha through gsplat, allowing every observed Gaussian to receive a
gradient even if it was hidden under full-scene transmittance. Base geometry
and opacity remain frozen. No source assets are introduced at query time.

This improves actual subset diagnostics in all three evaluated input conditions
over the matched visible-membership objective; normalized membership-image
metrics can regress. Keep both output types explicit. This is one-scene
development evidence with SAM region targets, not a five-task promotion.
See the [complete experiment ledger](../experiments/2026-10-07-query-preserving-gaussian-field-bootstrap.md).

## Two-scene evidence and retained boundary

The [next ledger](../experiments/2026-10-07-query-field-two-scene-and-readout.md)
adds fixed source-budget controls and actual LERF scoring. The current shared
member decoder fails to retain the exported field's native text selection
capability on both scenes. A native-only bypass is retained as a diagnostic;
it does not meet the shared-member decision above and is not a new default.
Likewise, single-anchor selection is not an equivalent solution to genuine
category unions. No additional teacher, source-query fallback or label-trained
member component is introduced to disguise this gap.

Keep geometry and native coordinates protected. The next bounded mechanism
study should compare reliable registered and inferred native support under the
same source budget and operator, measuring both identity and actual extent.
Compression awaits a stronger matched high-capacity reference. This is a
refinement of the implementation priorities, not a relaxation of the accepted
architecture or a claim that the full research direction has failed.

## Training-supported inferred inputs

The [three-scene support control](../experiments/2026-10-07-training-supported-identity.md)
identifies a concrete input-domain mismatch: original native peaks frequently
have negligible training exposure and are not eligible member-training supports.
Allow a frozen train-only eligibility mask for inferred anchors. This condition
is derived from the original reliable SAM interiors/exclusions, never benchmark
queries or development masks. The observed output carrier remains complete;
explicit registered inputs remain unchanged, including inputs outside this mask.

The fixed local gate passes on figurines, ramen and a newly constructed teatime
field without changing native, group or geometry tensors. The third scene is
historically opened development evidence, not independent final testing. Keep
this as an opt-in supported-input policy. It does not establish full identity
correctness, complete member extent, compactness or five-benchmark replacement.
