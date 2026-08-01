# VALA ScanNet OVS exact-protocol reproduction (2026-08-01)

## Protocol

- `paper8`: `scene0000_00`, `scene0062_00`, `scene0070_00`,
  `scene0097_00`, `scene0140_00`, `scene0347_00`, `scene0400_00`,
  `scene0590_00`.
- `code9`: `paper8` plus `scene0645_00`. The VALA paper specifies eight
  scenes; the current repository script contains nine, so the ninth scene is
  reported only as a cohort-sensitivity check.
- Text-conditioned fixed-vocabulary Gaussian classification with the official
  19/15/10 ScanNet class-name sets and CLIP text embeddings.
- Full-resolution semantic lifting with gsplat marginal-contribution
  significance (`alpha * T`) and the released robust aggregation rule.
- Released Gaussian pseudo-GT construction: anisotropic Mahalanobis density,
  Euclidean radius `5 * max(scale)`, candidate top-k 1000, class-balanced vote,
  and the released empty-candidate fallback.
- Gaussian metric weight: `sigmoid(opacity) * scale_x * scale_y * scale_z`;
  metrics are averaged over GT-present classes within each scene and then
  equally over scenes.

## Results

All entries are `mIoU / mAcc` in percent.

| Cohort | 19 classes | 15 classes | 10 classes |
|---|---:|---:|---:|
| VALA paper (`paper8`) | 32.11 / 50.05 | 35.10 / 54.77 | 46.21 / 65.61 |
| Local exact protocol (`paper8`) | **34.53 / 51.59** | **37.96 / 56.77** | **47.36 / 67.47** |
| Added `scene0645_00` only | 30.15 / 50.32 | 35.28 / 56.74 | 37.05 / 57.53 |
| Local exact protocol (`code9`) | 34.04 / 51.45 | 37.66 / 56.77 | 46.22 / 66.36 |

The old resolution-2/proxy-significance features, re-evaluated with the exact
Gaussian pseudo-GT, score 12.48/24.07, 16.09/28.87, and 24.51/43.12. Thus the
pseudo-GT correction is necessary but explains only about 0.5--1.8 mIoU of the
old gap. Full resolution, exact per-view significance, and official robust
aggregation are the dominant corrections.

## Evidence paths

- Exact `paper8` features and evaluation:
  `/mnt/pool/sqy/results/RADIO-GS/output/protocol_audit_20260801/vala/scannet_official_significance_paper8_v2`
  (`gaussian_eval/vala_scannet_gaussian_protocol_results.json` SHA-256
  `81e6584a29eab59ffacba91b21d079c88213643a1b9a234356240d70b7d13740`).
- Ninth-scene sensitivity result:
  `/mnt/pool/sqy/results/RADIO-GS/output/protocol_audit_20260801/vala/scannet_official_significance_code9_scene0645_v1`
- Old-feature pseudo-GT ablation:
  `/mnt/pool/sqy/results/RADIO-GS/output/protocol_audit_20260801/vala/scannet_proxy_features_gaussian_paper8_v1`
- GPU0 telemetry:
  `/mnt/pool/sqy/results/RADIO-GS/output/protocol_audit_20260801/telemetry/vala_scannet_paper8_v2_gpu0.csv`

## Scope caveat

This is a released-code semantic-lifting and evaluation-protocol reproduction
using the available local RGB Gaussian geometry. It is sufficient to isolate
and close the evaluation-protocol discrepancy, but it is not a fresh end-to-end
30k-iteration VALA geometry reproduction.
