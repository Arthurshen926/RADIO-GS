# Separate common grounding from relational text reasoning

Status: accepted for ScanNet-UQIS-9 v0.2.

UQIS v0.2 scores a `Unified-Query Core Cohort` on all four modalities and
reports `UQ-Rank` plus dev-calibrated `UQ-Mask`.  Core text expressions must
identify the instance without spatial, comparative, ordinal, or negative
relations.  Targets that require those operations remain in the full cohort
and are reported as a separate `Relational Text Challenge`; they do not enter
the common-modality core aggregate.  This keeps the primary comparison about
query-to-field grounding while retaining relational language as an explicit
system-level challenge instead of deleting difficult examples or allowing it
to dominate only the text arm.
