# ADR0024: Include our v10 mechanisms in the paper; test NVOS evidence loss

Date: 2026-10-17. Status: accepted scope; new NVOS candidate not promoted.
Supersedes ADR0023's exclusive paper framing, not its recorded experiments.

## Decision and reason

The user explicitly clarifies that v10 is our method, including its high scores
and successful design. Treating it only as an external-like frozen provider has
discarded the strongest completed contributions and narrowed the paper to a
small optional selector gain. Correct this attribution without changing scores.

The paper studies **query-conditioned evidence fusion for 3D responses**:
identify applicable query evidence, distinguish support/exclusion/unknown, and
combine it through geometry while preserving complete task outputs. v10 is the
implemented system. Its native matching, identity admission, source fusion and
complete membership are in-house components. External teachers and comparison
methods remain attributed to their authors. Freezing a component for a controlled
experiment does not change its ownership.

This is one evidence-to-response problem with typed realizations, not a claim
that every query executes the same network or solver. Native similarities,
reference tracks, categorical competition and conditional membership retain their
mathematical differences. A common abstraction does not prove a universally
superior algorithm. Weighted averaging is not claimed as a new primitive.
Historical controls support exactly the components they change; their gains
are non-additive and not L11 gains.

ADR0023's complete-response replacement remains a separately measured optional
extension with matched controls and unfinished LERF/NVOS transfer. It no longer
determines whether the underlying five-task v10 method is reportable. Do not
stitch L11, NVOS candidates and historical task maxima into a purportedly
evaluated improved configuration.

## NVOS intervention

Test whether low-resolution binary erosion discards valid source evidence.
Keep all eight tasks, original positive prompts, SAM3 tracks, frozen geometry,
source access, solver iterations and the 0.5 output threshold. Cross source
resolution (256x192 / 512x384) with old eroded evidence / confidence-weighted area
coverage. Scale the legacy erosion radius to preserve its normalized footprint.
The candidate pools native probability times confidence and confidence separately
before dividing. Confidence is not calibrated teacher correctness.

All predictions precede scoring. Score the retained and LUDVIG GT conventions
separately. The cohort is already development data; sealing predictions does
not make it an independent final test. No default promotion, per-task winner
selection or GT-chosen thresholds. Other benchmark results are unchanged
references bound by hashes, not fresh inference.

The first GPU attempt exceeded its explicit memory cap before scoring. Preserve
its failure receipt and sources. The CPU CSR solver must reproduce the retained
low-resolution response before accepting the factorial comparison. Dense arrays
remain transient; small receipts, scores, source snapshots and reports persist.

## Paper organization

Problem and related work; implemented method; inputs and metrics; five-task
comparisons with v10 labelled Ours; matched historical component controls; NVOS
controlled study; optional replacement diagnostics; limitations. Reuse verified
external reproductions. Ownership, controlled accuracy contribution and novelty
are distinct questions. Relabelling v10 creates neither a new score nor a new
isolated causal gain.

## Completed observation (2026-10-17)

N01 completes all eight tasks: legacy-low / coverage-low / legacy-high /
coverage-high IoU is 86.5134 / 86.7069 / 85.6138 / 87.7776. The combined change
improves seven tasks; mean AP rises 98.0222 -> 98.1527 but orchids AP decreases.
It remains a development candidate, without changing the default. A17/A19 show
the original independent set obtains 85.2245 IoU and is not a hidden better
NVOS aggregate. Crossed fields/readouts demonstrate that silhouette-trained
membership cannot simply replace normalized semantic response scores. Preserve
objective/readout compatibility as part of functional inheritance.
