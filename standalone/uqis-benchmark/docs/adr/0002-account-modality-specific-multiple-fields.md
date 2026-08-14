# Account modality-specific multiple fields as one method system

Status: accepted for ScanNet-UQIS-9 v0.1; field-assignment rule superseded by
ADR 0005 for v0.2.

## Context

Some comparators, including LUDVIG, do not encode text and visual interaction
in one persistent feature field. Excluding them would remove useful
modality-level comparisons, while calling their collection of fields a
universal field would erase the main representation distinction and its
storage cost.

## Decision

UQIS distinguishes a `single_universal_field` from a
`modality_specific_multi_field` method system. A complete method system may
seal all four modalities and receive the aggregate system metric, but its row
must retain its representation scope. Every modality maps to exactly one
content-bound field, and every per-scene field artifact and mapping receipt is
inventoried. Field count and the sum of persistent per-scene field bytes are
reported; they may not be normalized to one field or charged only at the
largest field.

For the LUDVIG comparator, each scene has two field families: a CLIP language
field for text and a DINOv2 visual field for image, registered 2-D point, and
world 3-D point queries. The strict prompt adapters may consume only the
frozen field and authorized prompt input; they may not use query-time captured
RGB or the historical SAM path.

## Consequences

LUDVIG can be evaluated as a complete multimodal method system without being
misrepresented as a universal representation. Its aggregate is comparable as
a system response score, while single-field claims and storage-efficiency
comparisons remain separately identifiable. Missing modality assignments,
unbound mapping receipts, or understated storage fail validation.
