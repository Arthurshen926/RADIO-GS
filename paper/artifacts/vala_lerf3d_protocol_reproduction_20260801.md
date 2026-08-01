# VALA LERF-3D protocol reproduction (2026-08-01)

This reproduction runs VALA's released semantic pipeline and evaluator on the
four LERF-OVS scenes while reusing compatible iteration-30000 OccamLGS RGB
Gaussian geometry. It isolates the semantic lifting and evaluation protocol;
it is not a fresh end-to-end VALA RGB-geometry training reproduction.

## Exact protocol

- The annotation frames are excluded from language-feature lifting but remain
  available as RGB geometry-training views, matching the released LERF intent.
- VALA's COLMAP reader compares extensionless camera stems with
  `sparse/0/test.txt`. Therefore the test file must contain `frame_00006`, not
  `frame_00006.jpg`. A suffix-bearing Occam-style list silently matches zero
  cameras and leaks every annotated view into semantic lifting.
- Each of three feature levels uses the official gsplat marginal-contribution
  significance and stochastic robust aggregation (`tau_mass=0.75`,
  `tau_abs=0.13`).
- For each text query, the level with the highest raw relevance peak is chosen.
  Relevance is smoothed with KNN-10, min/max mapped and clipped, thresholded at
  0.6 in 3D, and only the selected Gaussians are alpha-rendered into the
  annotated views.
- mIoU and Acc@0.25 are computed per annotated object and then averaged within
  each scene; the headline result is the equal mean over four scenes.

The validated train/test view counts are:

| Scene | Language-lifting train views | Held-out annotation views |
|---|---:|---:|
| `figurines` | 295 | 4 |
| `ramen` | 124 | 7 |
| `teatime` | 171 | 6 |
| `waldo_kitchen` | 182 | 5 |

## Results

All entries are `mIoU / Acc@0.25` in percent.

| Scene | Released VALA semantic pipeline |
|---|---:|
| `figurines` | 58.35 / 87.50 |
| `ramen` | 44.03 / 66.20 |
| `teatime` | 69.44 / 86.44 |
| `waldo_kitchen` | 44.68 / 77.27 |
| **Scene-equal mean** | **54.12 / 79.35** |

The VALA paper reports 43.29/64.30. The local released semantic pipeline is
therefore +10.83/+15.05 above the paper result even without retraining VALA RGB
geometry. In contrast, the previous local 32.53/50.43 result used a fixed L3,
threshold 0.4, and compatibility features rather than VALA's per-query
three-level selection and 0.6 direct-3D protocol. It must not be interpreted as
evidence that the method itself is intrinsically weak.

## Evidence and scope

- Combined official evaluator result:
  `/mnt/pool/sqy/results/RADIO-GS/output/protocol_audit_20260801/vala/lerf3d_occam_geometry_v1/evaluation/all_metrics_30000_0.6.json`
  (SHA-256 `1a4a4ce2856b2af7e83d0c157aebf3498d1740ecc5deaefc39403c0daa01fcae`).
- Split audit:
  `/mnt/pool/sqy/results/RADIO-GS/output/protocol_audit_20260801/vala/lerf3d_occam_geometry_v1/split_audit.json`
  (SHA-256 `120651911b2a2f034b5ffe6a59d7e7a50986f0ee921b0537ba5d7800e2b8bf25`).
- Staged data, feature checkpoints, masks, GT masks, and telemetry:
  `/mnt/pool/sqy/results/RADIO-GS/output/protocol_audit_20260801/vala/lerf3d_occam_geometry_v1`
- Clean VALA commit: `48902a541333d65aeb0aebf64ad664777a27c3fc`.
- Official VALA gsplat fork was used; generic gsplat 1.4 lacks the required
  `activated`/`significance` outputs.
- GPU0 peak over the successful lifting/readout jobs: 61 C, 294.03 W,
  15,768 MiB. No thermal pause, PCIe loss, or driver failure occurred.

Because the exact semantic pipeline already exceeds the paper, fresh VALA RGB
training is not required to diagnose or close this evaluation-protocol gap. It
would be necessary only if an end-to-end, asset-identical VALA reproduction is
later required for publication provenance.
