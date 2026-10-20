# TMM study normalization — 2026-10-16

## Outcome

The active entry points now agree on one query-response study over frozen
providers. `configs/research/tmm_response.json` registers E01–E04 as four complete
studies, F01–F03 as interrupted attempts and A01–A05 as compatibility/attribution
audits. No numerical inference defaults, benchmark outputs or protected assets
were changed. This work adds no accuracy result.

`python -m radio_gs.response_study` supplies status, verification, metric export,
artifact refresh, paper build, targeted tests and bounded replay planning.
Numerical kernels stay at their original audited paths. Report verification
checks the complete 31-job input contracts, 124 sealed prediction reports,
source-scene/UQIS aggregation, 42 historical evidence files, baseline consistency
and method/code bindings. It does not load prediction tensors or independently
re-score ground truth.

`paper/tmm_response/artifacts/metrics.csv` contains 128 candidate metric rows
(eight arms × sixteen metrics), with baseline, effect, descriptive intervals,
input access, frozen sets and actual-intervention status. There is no assembled
best-per-task method. `study_catalog.json` includes the source-version ledger.

## Corrected method attribution

Inspection of all nine effective UQIS map specifications and the runtime caller
confirms that image-estimated anchors use `DirectAnchorMembership.infer_region`.
`QueryConditionedMembership` is an optional candidate absent from these maps.
The retained `infer_region` calls `infer`; its estimated anchor is forced to
probability one even though metadata declares identity uncertainty. A regression
test demonstrates this inherited behavior using a classifier that otherwise
returns low membership probability for every element.

The manuscript and generated method ledger now describe this behavior. The
desired separation of estimated-anchor and actual-click authority remains a
research gap. Its impact on benchmark accuracy has not been isolated by this
audit, and no runtime change was made to fit the earlier theoretical claim.

## Reproduction limits and storage

E01 has eight unavailable source-version receipts, representing five distinct
versions across four files. Hashes alone cannot recover the source. Results and
report seals remain checkable, but exact historical inference replay is not
certified. E02 retains a matching source snapshot. The registered prediction
sources for E03/E04 match current code; fresh scoring requires recomputing the
released transient predictions and still depends on the original input assets.

The entry point disables E01/E02's historical persistent writers and only plans
full-cohort E03/E04 replays under new directory names with matching source hashes.
The existing bounded runners remain responsible for storage enforcement. The
study remains at 2,042,518,591 logical bytes under its 2 GiB limit. No new dense
arrays were created during normalization.

## Documentation and validation

Root README shrank from 563 to 52 lines; CONTEXT from 3349 to 109 lines; the paper
guide from 265 to 131 lines. Exact original text and SHA-256 receipts are retained
under `docs/history/2026-10-16-normalization-snapshots.json`. Current instructions
are separated from historical continuation notes.

All 56 targeted tests passed. Standard-library-only verification also passed
with Python site packages disabled. Both E03/E04 replay plans were checked
without launching jobs. Main and diagnostics PDFs each compile to six pages
without LaTeX warnings or unresolved references; the corrected anchor limitation
is present in the rendered main paper. Local documentation links and archived
text hashes were checked.

Reproducible commands and limitations are in `paper/tmm_response/README.md`.
The validation receipt is
`paper/tmm_response/artifacts/normalization_validation.json`; the PDF receipt is
`paper/tmm_response/method_validation.json`. A single transferable admission rule,
real UQIS mechanism intervention and independent validation remain open.
