# VALA official-code anchor

Checkout and revision are in upstream.lock.json. The next47 figurines pilot uses the unmodified official extractor with its gsplat fork and existing historical RGB Gaussians converted to the official checkpoint container. It is **not** a retrained author-protocol result or a same-input comparison to RADIO-GS. Existing geometry training split and language-cache provenance still need to be established before formal comparison.

Artifacts: `/root/RADIO-GS/output/vala_official_next47`. The isolated source overlay links the original RGB, COLMAP and language data read-only; `sparse/0/test.txt` contains annotation frame **stems**, matching the official camera loader. No benchmark mask contents are used to generate features.

Entry point: `/root/baselines/VALA/gaussian_feature_extractor.py --iteration 30000 --feature_level LEVEL --use_efficient --eval`, with `PYTHONPATH=/root/baselines/VALA/submodules/gsplat:/root/baselines/VALA` and explicit source/model directories. Levels 1, 2 and 3 are complete, using 295 source cameras and 4 held-out rendering cameras. Official `eval.render_lerf_by_text_langsplat` and `eval.compute_lerf_iou` completed at mask threshold 0.6: figurines mIoU 57.53%, Acc@0.25 89.29%, Acc@0.5 64.29% (56 annotated query-view pairs).

This is a one-scene official-code pilot, not a full four-scene reproduction. The official evaluator thresholds saved alpha PNGs and JPEG ground-truth masks at byte value 10; these results must not be substituted for RADIO-GS probability-map IoU, AP or localization. The source-camera budget, RGB geometry and semantic caches differ from RADIO-GS. Logs: `figurines/render.log` and `figurines/evaluation.log`. Runtime dependencies added: open_clip_torch 2.24.0 and ftfy 6.3.1; official code unchanged.

Next48 completed all four LERF scenes through the official three-level code:
`python scripts/research/vala_exchange_next48.py`. Three controlled conditions
are historical RGB geometry with all non-annotation source views, historical RGB
geometry with the current source subset, and current MoGe geometry with that same
subset. Actual subset counts are20/25/23/22 (the arm suffix20 is shorthand).
Selected-projection macro IoU is51.854/49.527/35.187 respectively. These reuse
historical feature caches and geometry; they are not a from-scratch reproduction
of the published score or an equal geometry-training budget comparison.

The wrapper changes only source-camera selection and the supplied geometry;
upstream fusion, scale choice, smoothing and mask scoring are unchanged. A
separate source-RGB MoGe carrier fit has mixed two-scene results and is not a
replacement. Full results: `output/vala_exchange_next48/summary.json` and
`docs/experiments/2026-09-29-carrier-and-set-decoding-next48.md`.
