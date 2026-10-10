# Exact partial-relation union before lifting

Four LERF scenes, 68 texts, unchanged 72 source views/5137 regions, existing
teacher_acceptance_v1 binary acceptances and mass>0 filter. No new teacher calls,
training, target-label access or threshold selection. All predictions before scoring.
Five controls: admission, teacher_protected, teacher_union_protected (exact cached),
atom_parts, atom_protected (PRIMARY). Same prior, finite P/N integration and protection.

Compile query-independent overlap atoms from existing positive/negative/unknown
pixel relations. Positions with identical region relation vectors share an atom.
Aggregate the existing confidence-weighted lift over atoms, then at query time
OR accepted positive incidence and AND accepted negative incidence BEFORE lift.
Empty acceptance produces neither positive nor negative. This is exact union of
existing partial labels, not new morphology on the union of raw masks. Compile
before reading query acceptance; exhaustive single-region marginals vs existing
field must agree within2e-6. The extra state is a research artifact, not yet an
integrated deployment field; tensor bytes and atom count reported explicitly.

The runtime primitive reads only compiled factors and current binary acceptance;
this study still uses privileged cached source-teacher acceptance. Thus it remains
SOURCE-ASSISTED DIAGNOSTIC, not restored deployable performance or a full-five model.
No v10 query fitting, benchmark answer bank, new default or per-scene routing.
If preserving composition fails, do not claim it solves the actual-set objective.
Evaluate the joint member decoder separately, with these source/acceptance budgets
fixed. Standard inference must ultimately remain one field without source replay.
