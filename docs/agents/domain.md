# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root.
- **`CONTEXT-MAP.md`** at the repo root if it exists.
- **`docs/adr/`** — read ADRs that touch the area being changed.

If these files don't exist, proceed silently. The `/domain-modeling` skill creates them lazily when terms or decisions are resolved.

## File structure

This repository uses the single-context layout:

```
/
├── CONTEXT.md
├── docs/adr/
└── radio_gs/
```

## Use the glossary's vocabulary

When output names a domain concept—in an issue title, proposal, hypothesis, or test—use the term defined in `CONTEXT.md`.

If the required concept is absent, reconsider whether it is new terminology or record the gap for `/domain-modeling`.

## Flag ADR conflicts

If work contradicts an existing ADR, surface the conflict explicitly rather than silently overriding it.
