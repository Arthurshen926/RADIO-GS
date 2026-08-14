# ScanNet-UQIS-9 LUDVIG controlled result — 2026-08-14

## Claim boundary

This is a valid, evaluator-controlled, one-shot effectiveness evaluation of a
complete sealed LUDVIG method system on the nine-scene construction authority.
It is **not** a public formal leaderboard row: the public Evaluation Authority
and sandboxed formal release remain disabled. It is also a benchmark-local
LUDVIG adapter result, not an official LUDVIG paper reproduction and not
comparable to the historical paper metric.

The method is honestly reported as `modality_specific_multi_field`:

- one OpenCLIP 512-D text field per scene;
- one DINOv2/PCA40 field per scene shared by image, 2-D point and 3-D point;
- 18 persistent fields across nine scenes;
- 37,209,952,854 persistent bytes (34.66 GiB) charged in total.

## Sealed execution

- scenes: 9
- targets: 67
- queries: 268 (67 for each of text, image, point_2d and point_3d)
- every query ran in a fresh process and read-only one-query workspace;
- all outputs are finite `float32` probability vectors on the bound official
  ScanNet mesh domain;
- probability mapping was frozen sigmoid scale=1, bias=0;
- no threshold/calibration was fitted on evaluator labels;
- all predictions and the complete field inventory were sealed before the
  one-shot evaluator opened private instance labels.

Key identities:

- construction authority: `0000eae5408ac9bd6e2def450dec03ec40278f348cb951c5c9557acd6bffde76`
- LUDVIG method identity: `4af6d4d321e3a3ea12b931e333782225776b214222e34a9b8f85d2c2ad7b00a7`
- field inventory: `27065111ddb434b156b31e99a330d45a1454b2b2e39f58791b5c26a4292bd45a`
- sealed prediction batch file: `99cfa080bdbba0429629416cfba6afb786a8f24b404a196e30d7e6f55e8bb6b4`
- prediction inventory: `ce17cf444ee1685cd793b9d0e7bde21fba2548c5e762fa9b6bc6a167cf980678`
- result report: `4c4c77f703e93c6d7b629119b492297019f25724b7571773f91b36c09e8604a0`
- consumed ledger: `87159fa3f02f90c3d2227742e33cc0a4a0655e8e5cbcb3d550a5dc117449a50b`

## Metrics

All point estimates are equal-scene macro averages. Confidence intervals use
2,000 scene-clustered bootstrap samples with frozen seed `20260813`.

| Modality | AP | AP 95% CI | Oracle IoU | Fixed IoU@0.5 | Fixed IoU 95% CI | Centroid error (m) |
|---|---:|---:|---:|---:|---:|---:|
| text | 0.1116 | [0.0793, 0.1464] | 0.1011 | 0.0533 | [0.0341, 0.0750] | 1.8841 |
| image | 0.2395 | [0.1743, 0.3092] | 0.1944 | 0.0463 | [0.0350, 0.0587] | 1.9958 |
| point_2d | 0.5159 | [0.4709, 0.5656] | 0.3824 | 0.0504 | [0.0378, 0.0631] | 1.7288 |
| point_3d | 0.5179 | [0.4678, 0.5706] | 0.3846 | 0.0481 | [0.0364, 0.0604] | 1.7224 |

The complete-system UQ-Mean (equal-modality mean of scene-macro fixed
IoU@0.5) is **0.04953**, with scene-clustered 95% CI
**[0.03729, 0.06157]**.

Per-scene UQ-Mean ranges from 0.01864 (`scene0700_00`) to 0.08124
(`scene0203_00`).

## Interpretation

The benchmark is operational and discriminative rather than returning a
constant/plumbing-only response. The registered 2-D and 3-D prompt paths have
nearly identical AP and oracle IoU, which supports the paired prompt
construction and the RGB-free 2-D rendering interaction. Image provides
moderate ranking signal, while the separate CLIP text field is substantially
weaker on these Nr3D expressions.

The large gap between prompt AP/oracle IoU and fixed IoU@0.5 isolates the main
LUDVIG integration weakness: the frozen sigmoid is not an empirically fitted
probability calibration, and it produces overly broad selections. This test
cohort must not be used to repair that threshold. A future formal row needs a
separate frozen dev cohort and dev-only calibration receipt, followed by a new
evaluation-authority release with secret query identifiers and true sandbox
runtime receipts.

## Immutable evaluator artifacts

- result:
  `/mnt/pool/sqy/results/RADIO-GS/output/scannet_uqis_9_v0_1/ludvig_evaluation_v1/controlled_result_v1.json`
- one-shot ledger:
  `/mnt/pool/sqy/results/RADIO-GS/output/scannet_uqis_9_v0_1/ludvig_evaluation_v1/controlled_evaluation_ledger_v1.json`
- prediction seal:
  `/mnt/pool/sqy/results/RADIO-GS/output/scannet_uqis_9_v0_1/ludvig_evaluation_v1/sealed_prediction_batch.json`
- method/field seal:
  `/mnt/pool/sqy/results/RADIO-GS/output/scannet_uqis_9_v0_1/ludvig_evaluation_v1/method_seal_v1/`

