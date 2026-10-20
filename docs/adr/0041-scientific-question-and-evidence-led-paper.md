# ADR0041: Organize the paper around partial-evidence authority

2026-10-18. Accepted for manuscript organization; complements ADR0024.
No inference defaults, experiment summaries, thresholds or weights change.

The user prioritizes a scientifically coherent TMM paper over more incremental
NVOS tuning. The manuscript must explain one problem and one central mechanism:
how local query evidence can authorize support or exclusion and form a complete
3D response. Query applicability, spatial scope and output-compatible fusion
are linked decisions within that mechanism. Native features and dimensions are
outside the research scope. v10 remains our method; reference completion is
an explicitly declared extension, not an input-identical replacement baseline.

Add a problem/analysis section with (1) the ambiguity of complete extent under
partial binary constraints, (2) exact attenuation by false negative mass for
fixed-weight direct consensus, and (3) a compositing counterexample showing that
semantic rendering and actual selection are different operations. Give proofs
and assumptions. These are explanatory results, not performance guarantees,
a new convergence theorem, or proof that estimated completion is correct.

Organize experiments as protocol and five-output comparisons, followed by:
identity before response formation; reference scope before transport; and
stronger fitting/readout as competing explanations. Promote the N12 and N15
paired original/completed-reference contrasts as the scope evidence. Preserve
the source-only N15 result, N14 primary and secondary policies, and N16's higher
IoU/lower AP tradeoff. Do not turn a secondary or diagnostic arm into a
predeclared primary after seeing results. Historical component contrasts are
not final-system leave-one-out ablations and cannot be added together.

Main tables are explicit editorial views of the complete verified catalog.
Keep all external comparison rows and their provenance, all historical NVOS
variants in the diagnostic supplement, and L11 with its supervision/controls
in a separate optional supplement. No per-scene winners or altered numbers.
Theory, abstract, introduction, experiments, discussion, figures and main-table
generation must be covered by the method provenance manifest.

A small source-only snapshot preserves the pre-reorganization manuscript.
No benchmark inference or large experiment outputs are needed for this change.
Further research priority is frozen-rule validation and matched prompt/teacher
capability attribution, not reopening the stopped N15/N16 searches. Universal
cross-teacher admission, unseen-scene generalization and same-protocol SOTA
remain unestablished.
