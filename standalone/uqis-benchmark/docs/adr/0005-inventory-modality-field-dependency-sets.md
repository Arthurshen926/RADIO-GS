# Inventory modality field dependency sets

Status: accepted for ScanNet-UQIS-9 v0.2.

Each modality maps to a non-empty, content-bound `Method Field Dependency
Set`, not necessarily one field.  The method inventory charges the per-scene
union of every dependency exactly once and rejects undeclared query-time field
selection.  This permits an honestly accounted LUDVIG text comparator to use
both its CLIP relevance field and DINO graph field while preserving the
distinction from a `single_universal_field` and preventing auxiliary semantic
fields from becoming free hidden state.
