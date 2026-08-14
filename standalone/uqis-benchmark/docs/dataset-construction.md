# Dataset construction

## Required upstream assets

UQIS construction requires users to obtain these assets themselves:

- ScanNet v2 `.sens` streams, official meshes, aggregation JSON, segmentation
  JSON, and exported color/depth/instance/label/pose frames;
- ReferIt3D's Nr3D annotation table.

ReferIt3D is the project/dataset family. Nr3D is the human free-form referring
expression subset used by UQIS. They should not be described as two unrelated
annotation sources.

Every input file is content-bound by SHA-256 in a private construction receipt.
No method prediction or evaluator metric is available during construction.

## Frozen nine-scene cohort

The current test candidate uses:

`scene0030_00`, `scene0203_00`, `scene0246_00`, `scene0249_00`,
`scene0353_00`, `scene0435_00`, `scene0535_00`, `scene0700_00`, and
`scene0704_00`.

The user accepted nine scenes without mining the full ScanNet validation split
for a tenth. Rejected and replacement candidates remain in the cohort
derivation ledger with model-independent reasons.

## Target eligibility

The constructor recomputes all facts from bound assets. Important v0.1/v0.2
constants include:

- 6–8 targets per scene and at most two selected targets of one raw class;
- at least four semantic categories and three same-class-distractor targets;
- at least 500 official mesh vertices per target;
- query projection at least 1,000 pixels **or** 1% of the image;
- projection purity at least 0.90;
- legal mapping visibility in at least five sampled frames;
- mapping-surface coverage at least 0.70 within 5 cm;
- at most three shared query frames per scene.

Structural wall/floor/ceiling objects are excluded. The constructor verifies
2-D instance projections against official 3-D instance vertices before using
them.

## Text selection

An eligible Nr3D row must:

- refer to the official target object;
- be marked correct and mention the target class;
- pass the frozen view-independence rule;
- contain 2–64 whitespace-delimited tokens;
- have explicit `uses_spatial_lang` evidence in the v0.2 profile.

For Core, deterministic selection is restricted to eligible non-spatial rows
and then chooses the minimum stable annotation ID. If no non-spatial row
exists, a valid relational row is retained in the challenge tier.

## Image and point pairing

For every target, the selected shared query frame maximizes valid target pixels
under the scene-level frame budget. The image crop uses a 15% padded target
mask box, is normalized to 224×224 RGB, stripped of source metadata and saved
under an opaque query identity.

The positive 2-D click is selected in the interior of the target projection,
aligned with depth, and unprojected through the frozen camera matrices. The
resulting world coordinate is the point-3D query. The release validator
reprojects it and fails if the paired points disagree beyond numerical
tolerance.

## Query-union frame exclusion

Mapping may not see any query frame. For all selected query frames in a scene,
the constructor removes the union of:

- frames within ±5 positions in the full `.sens` order;
- poses within 10 cm translation and 8 degrees rotation.

Coverage is recomputed after this union exclusion. A field construction receipt
must bind the actual mapping-observation inventory, not merely repeat caller
supplied frame IDs.

## Construction outputs

Private construction produces scene records, target records, normalized crops,
official mesh arrays and derivation receipts. Freezing then splits them into:

- public scene-domain manifest;
- four physically separable method query bundles;
- mapping-only exclusion manifest;
- aggregate-only public cohort summary;
- evaluator-private target pairing and instance labels.
