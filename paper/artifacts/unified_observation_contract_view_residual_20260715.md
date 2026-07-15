# Unified observation lifting and view residual — 2026-07-15

## Method decision

The canonical method now has one versioned, dataset-independent observation
lifting policy, `canonical-mpr-v1`:

- deterministic uniform temporal sampling, at most 120 available training views;
- `raster_gaussian_top1` assignment;
- `alpha_depth` registration responsibility;
- `contribution_mean` fusion across views;
- per-view L2 normalization;
- absolute/relative depth tolerance 0.08/0.02 and alpha threshold 0.02;
- official capability projection per 2D view before MPR;
- one exact, feature-independent responsibility sidecar shared by raw RADIO,
  DINO, and SAM teacher spaces.

Resolution, camera calibration, number of available frames, and held-out frame
IDs remain dataset provenance.  They are not method hyperparameters.  New MPR
builds declare the complete contract and its SHA-256 digest; new canonical-field
training fails closed if it is missing or differs.  Old caches can only be used
through explicit compatibility certification that checks every policy field.

## Existing-field audit

| Field | Status | Difference from canonical-v1 |
|---|---|---|
| LERF Figurines | not certified | legacy metadata declared normalization, but the raster implementation did not apply it |
| LERF Teatime | not certified | legacy metadata declared normalization, but the raster implementation did not apply it |
| LERF Waldo Kitchen | not certified | legacy metadata declared normalization, but the raster implementation did not apply it |
| ScanNet scene0000 | incompatible | 279 views, `view_mean`, no per-view normalization |
| NVOS fern | incompatible | `view_mean`, no per-view normalization |
| SPIn-NeRF fern | incompatible | `view_mean`, no per-view normalization |

The legacy LERF fields and the incompatible ScanNet/NVOS/SPIn checkpoints remain
historical baselines and must not be described as certified unified-contract
results.  A later implementation audit found that `normalize_each_view=true`
was metadata-only in the raster branch.  The raster implementation now applies
L2 normalization to the per-pixel feature map before lifting and records
`per_view_normalization_applied=true`; certification requires this execution
evidence rather than the old intent flag alone.

## Low-capacity view residual

The canonical coefficient of primitive `i` is unchanged.  Only 2D rendering
adds

`delta_i(v) = (u_i * ((v - mean_i) A)) B * gate_i * scale`.

The training-observation weighted mean is exactly zero because `mean_i` uses
the exact replayed alpha-depth MPR weights and every operation after centering
is linear.  Rank is capped at 8 for a 256-D coefficient field: per-primitive
capacity is 3.125% of a dense view-conditioned coefficient.  The residual is
not loaded by primitive-domain text, image, registered-prompt, or world-point
query compilers.

Training is query-free.  It uses raw RADIO reconstruction plus frozen official
DINO/SAM dense alignment and local-affinity losses.  Checkpoint selection uses
non-benchmark validation frames and requires raw/DINO/SAM non-inferiority; no
text, mask, category, scribble, or point annotation is opened.

Both Ramen and Figurines replayed all 120 MPR views with exact per-row counts:
zero mismatch rows, zero validity flips.

## Cross-scene held-out results

### Ramen, four frames never used for MPR or residual selection

| Space | Base mean | + residual mean | Base p05 | + residual p05 |
|---|---:|---:|---:|---:|
| raw RADIO | 0.737653 | **0.755265** | 0.470433 | **0.521074** |
| official DINO | 0.828974 | **0.844005** | 0.613813 | **0.647048** |
| official SAM3 | 0.696469 | **0.702973** | 0.298704 | **0.315636** |

### Figurines, frozen architecture/loss/gates, three held-out frames

| Space | Base mean | + residual mean | Base p05 | + residual p05 |
|---|---:|---:|---:|---:|
| raw RADIO | 0.659131 | **0.679605** | 0.406109 | **0.441518** |
| official DINO | 0.780202 | **0.797006** | 0.565732 | **0.596494** |
| official SAM3 | 0.644578 | **0.651620** | 0.318157 | **0.330416** |

Dense and lower-tail fidelity improve in all six cross-scene comparisons.
However, SAM boundary-margin retention is essentially unchanged: Ramen
0.037447 -> 0.037007 and Figurines 0.086984 -> 0.086993.  The residual is
therefore retained as a view-context fidelity module, not claimed as a complete
semantic-boundary solution.

## Artifacts and verification

- contract audit: `output/optimization_20260715/unified_mpr_view_residual/contract_audit.json`;
- Ramen residual and audits: `output/optimization_20260715/unified_mpr_view_residual/ramen/`;
- Figurines residual and audits: `output/optimization_20260715/unified_mpr_view_residual/figurines/`;
- focused contract, residual, MPR, field, capability, and compositor tests:
  **57 passed**; `git diff --check` passes.
