# LangSplatV2 LERF-2D final LocAcc gap audit

Date: 2026-08-01  
Mode: CPU only; no checkpoint render, training, network access, or GPU use

## Decision

The exact-camera local row remains a valid **released-code-intent / local-
checkpoint compatibility diagnostic**, but it is not a strict reproduction of
the published LangSplatV2 row. After auditing the query strings, GT merge,
camera identity, metric denominator, OpenCLIP prompts, top-k, semantic-level
selection, localization filtering, bbox rule, and headline arithmetic, there
is no known remaining evaluator-protocol defect that plausibly explains the
whole `84.10 - 79.8077 = 4.2923` point LocAcc gap.

The dominant unresolved variable is checkpoint/training/feature provenance.
The released `train.sh` omits `--eval`, and `ModelParams` defaults to
`eval=False`, so the released training command uses all cameras. All twelve
local level checkpoints instead record `eval=True`, which with LLFF holdout
removes every eighth camera from language-field training. The local cohort also
starts from Occam-compatible RGB checkpoints and compatibility language
features rather than an identified official LangSplatV2 checkpoint bundle.

## What “Mean” can and cannot mean in the source-context table

The locally pinned paper table labels its baseline rows as published values
reported by OpenGaFF Table 1 and explicitly warns that it is not a strict
same-protocol ranking. Arithmetic over the printed LangSplatV2 scene cells
gives:

- mIoU scene macro: `(56.4 + 72.2 + 51.8 + 59.1) / 4 = 59.875`, which prints
  as `59.9`;
- LocAcc scene macro: `86.375`, which does **not** print as `84.1`;
- LocAcc weighted by the local/released evaluator denominators
  `56 / 59 / 71 / 22`: `84.1399`, which prints as `84.1`.

This is strong numerical evidence for a metric-specific matching rule for the
LangSplatV2 row: scene-macro mIoU and 208-query-weighted LocAcc. It is not
evidence for a benchmark-wide query-micro LocAcc convention. Every neighboring
2D baseline Acc mean matches its scene macro instead:

| Method | Printed Acc mean | Scene macro | 208-query weighted | Printed-value match |
|---|---:|---:|---:|---|
| LangSplat | 84.30 | 84.3000 | 81.7236 | scene macro |
| GAGS | 81.66 | 81.6575 | 79.3265 | scene macro |
| OccamLGS | 82.50 | 82.5250 | 82.2332 | scene macro |
| GOI | 59.20 | 59.2250 | 57.6707 | scene macro |
| GALA | 73.43 | 73.4275 | 72.1147 | scene macro |
| LangSplatV2 | 84.10 | 86.3750 | 84.1399 | query weighted |

Accordingly, reports should say **“the printed LangSplatV2 row numerically
matches mixed aggregation”**, not “LERF-2D formally uses mixed aggregation.” A
cached copy of the source paper's evaluator prose was not available for an
independent textual confirmation in this CPU-only audit.

## Query, GT, camera, and hit-count receipt

The raw annotations contain 228 object instances/bboxes. The released loader
merges 20 repeated same-category instances, leaving 208 frame-query evaluation
units. Query order exactly matches first occurrence in the annotation object
list, all bboxes are valid within the annotation dimensions, and exact camera
stems resolve all 22 labelled frames.

| Scene | Queries | Local hits | Paper LocAcc | Nearest paper hits | Approx. hit deficit | Query roles, train/test |
|---|---:|---:|---:|---:|---:|---:|
| Figurines | 56 | 46 | 82.1 | 46 | 0 | 26 / 30 |
| Teatime | 59 | 52 | 93.2 | 55 | 3 | 41 / 18 |
| Ramen | 71 | 51 | 74.7 | 53 | about 2 | 48 / 23 |
| Waldo Kitchen | 22 | 17 | 95.5 | 21 | 4 | 19 / 3 |
| **Total** | **208** | **166** | **84.1 overall** | **175** | **about 9** | **134 / 74** |

`166/208 = 79.807692%`. The nearest integer reconstruction of the four printed
scene rows is `175/208 = 84.134615%`, a nine-hit difference. Ramen deserves an
explicit caveat: no integer numerator over 71 lies within half of the printed
0.1-percent unit around 74.7%; the nearest value is `53/71 = 74.647887%`.
Therefore “nine hits” is the highest-information reconstruction, not an exact
paper-side hit receipt.

## Released evaluator contract checked at the pinned commit

| Axis | Released behavior | Local exact-camera run |
|---|---|---|
| Positive queries | exact annotation category strings, first-occurrence dict order | match |
| Text encoder | OpenCLIP ViT-B-16, `laion2b_s34b_b88k` | match |
| Negative prompts | `object`, `things`, `stuff`, `texture` | match |
| Effective quick-render top-k | hard-coded `4`; released shell also sets `4` | match |
| Localization activation | 29x29 average pool, `count_include_pad=False` | match |
| Localization level | argmax over each level's pooled peak | match |
| Peak ties | retain every coordinate equal to the selected maximum | match |
| Hit test | any tied peak in any merged bbox; boundaries inclusive | match |
| Mask threshold | not referenced by localization | `0.4` cannot alter LocAcc |
| Denominator | number of merged frame-category keys | 208, match |

The evaluator's CLI `--topk` value is unused in the quick path, but this is not
a mismatch here because both the hard-coded effective value and the released
shell value are four. The logged `chosen_lvl` lists are segmentation choices;
localization recomputes independent level choices and does not serialize them.

The audit checkout differs from the pinned commit in three files. The
`eval_lerf.py` delta is exact-camera selection plus disabled-by-default
visualization plumbing; the localization scoring body is unchanged.
`utils/loss_utils.py` contains a training-side float/epsilon compatibility
change, and `utils/vq_utils.py` contains an evaluation device-placement fix and
a training-side explicit-float32 quantizer change. The latter two are further
reasons the locally trained checkpoints are not strict paper assets.

## Why no further CPU counterfactual was run

The exact-camera output contains GT visualizations, four aggregate logs,
camera manifests, and cohort receipts, but no rendered language feature,
relevance map, localization peak, selected localization level, or per-query
hit bit. The twelve checkpoint directories likewise contain no reusable render
cache. Consequently a CPU-only top-k, localization-level, peak-tie, bbox, or
camera-role hit counterfactual cannot be computed from existing outputs.

A mask-threshold sweep would have zero information for LocAcc and would amount
to GT-driven tuning for mIoU, so it was deliberately not run.

## Executable conclusion

1. Keep `61.51%` scene-macro mIoU and `79.8077%` query-micro LocAcc only under
   the name “LangSplatV2 released-code-intent / local-checkpoint exact-camera
   diagnostic.”
2. Do not mark the row strict-table-eligible, and do not use its `-4.29` point
   delta as evidence of a remaining threshold or aggregation bug.
3. For a strict reproduction, prefer an identity-verified official pretrained
   checkpoint bundle. If unavailable, retrain from paper-equivalent RGB and
   language features with a clean pinned checkout and `eval=False`, then run
   exact-camera evaluation.
4. The next GPU evaluator pass should serialize, for every frame-query unit,
   the hit bit, camera role, selected localization level, peak score, and all
   tied peak coordinates. That is the minimum artifact needed to separate the
   remaining nine-hit difference by cause.

## Reproducibility

- CPU audit script:
  `radio_gs/scripts/audit_langsplatv2_lerf2d_final_gap.py`
- machine receipt:
  `output/protocol_audit_20260801/langsplatv2_lerf2d_final_gap_audit.json`
- receipt SHA-256:
  `0bda52d7f8598c731817d2a2e69e62a043e5ab2f16de91840ccecbd11c4543d2`
- targeted tests:
  `CUDA_VISIBLE_DEVICES='' bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_langsplatv2_lerf_protocol_audit.py`
  (`7 passed`)
