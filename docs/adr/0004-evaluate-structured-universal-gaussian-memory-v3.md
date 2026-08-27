# Evaluate Structured Universal Gaussian Memory v3

Status: Proposed

## Context

The Query-Native v2 experiments separated object identity from extent, but did
not demonstrate that complete co-membership and boundary topology were trained
into the persistent field. Historical readouts also allowed LERF 2D and 3D to
consume different object posteriors. The new candidate addresses this missing
mapping-time capability rather than extending a benchmark readout.

## Proposed decision

Evaluate **Structured Universal Gaussian Memory v3 (SUGM-v3)** under these
invariants:

- each Gaussian persists exactly one D512 latent and five reliability scalars;
- RADIO, DINO, SigLIP, and SAM are frozen mapping-time teachers and add no
  Gaussian-indexed feature sidecar;
- exact-MPR transports positive, negative, and unknown evidence but is not an
  object ground truth;
- differentiable rendering of one 3D membership posterior back to held-out
  source masks is the primary instance-capability gate;
- text, image, and prompt queries select identity anchors and then use the same
  instance projection and the same Gaussian posterior;
- strict LERF never opens target RGB, and 2D is only a rendering of the
  posterior evaluated in 3D;
- all method selection precedes benchmark evaluation and is bound to source
  authority, configuration, checkpoint, and method-contract hashes;
- historical readouts are accessible only through `radio_gs.v3.legacy_adapter`.

The first mandatory experiment compares a frozen-D512 projection arm, a
temporary per-Gaussian D16 instance-code oracle, and an instance capability
written back into D512. The D16 arm is an upper-bound diagnostic and is never
deployment eligible.

## Promotion gates

The D16 oracle must first improve scene-macro source-heldout mask IoU by at
least 0.05, reduce Brier score, improve boundary F, avoid per-scene regression,
and not increase unknown false-positive mass. Only then may a low-rank D512
residual be trained. Full benchmarks remain blocked until the D512 arm passes
the same source gate and fixed LERF/ScanNet/NVOS sentinels. LERF 2D and 3D must
both improve using one posterior without localization regression.

## Consequences

ADRs 0001--0003 and all corresponding results remain immutable baselines; they
do not supply implementation logic to v3. Failure of the D16 oracle redirects
work to source SAM authority, exact-MPR, geometry, or evaluation. Geometry
splitting is optional v3.1 work and cannot delay a valid v3.0 full evaluation.
