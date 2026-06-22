# Three-Task Open-Baseline Audit, 2026-05-17

Source recommendation file: `ChatGPT-RADIO模型多视角重建优化 (1).md`.

## Expert Request Parsed

The latest expert request changes the work from a single-paper polishing pass
into a stricter benchmark package:

| Track | Task | Required metrics | Current RADIO-GS/GaussFM evidence |
|---|---|---|---|
| T1 | LERF-OVS rendered-view open-vocabulary localization/segmentation | LocAcc, mIoU | GaussFM rendered feature maps: 0.8712 LocAcc / 0.5243 mIoU |
| T2 | LERF-OVS text-to-Gaussian direct 3D object selection | mIoU, Acc@0.25 | VPR fixed `thr0p25` + RGB snap: 0.4801 / 0.6760; SAM3-box strict readout: 0.5705 / 0.6835 |
| T3 | ScanNet-v2 open-vocabulary point-cloud semantic segmentation | mIoU, mAcc on 19/15/10 splits | RADIO-GS v67: 0.3538/0.6076, 0.3573/0.6203, 0.4293/0.7051 |

The paper should only claim a strict SOTA result where the same evaluator and
same data assets have been reproduced locally. Official-source rows remain
context rows until rerun through the local benchmark.

## Local Reproduction State

| Method | Local repository | Commit | State |
|---|---:|---:|---|
| OpenGaussian | `/root/baselines/OpenGaussian` | `1f99db1` | ScanNet T3 reproduced locally on 10 scenes. LERF T2 compatibility rerun completed on all four scenes with macro 0.4273 mIoU / 0.5865 Acc@0.25 / 0.4727 Acc@0.5. Provenance caveat remains: `language_features` came from the compatibility extraction path, not a strict official OpenGaussian extraction. |
| LangSplatV2 | `/root/baselines/LangSplatV2` | `1667303` | Public repo cloned with submodules. README requires LERF/3D-OVS/Mip-NeRF360 data plus pretrained output/checkpoints or a fresh training run. Local ABI blockers were cleared by rebuilding `simple_knn` and the feature-aware `diff_gaussian_rasterization` into `output/baselines/langsplatv2/local_site`; local compatibility reruns for `teatime`, `ramen`, and `figurines` trained all three feature levels to checkpoint 10000 and completed `eval_lerf.py --quick_render`. Current completed-scene summary: scene-mean 0.5810 LocAcc / 0.4282 mIoU, object-weighted 0.5860 LocAcc / 0.4360 mIoU over 186 queries. Provenance caveat remains: the runs start from local Occam-compatible RGB checkpoints and compatibility language features, not a strict released LangSplatV2 checkpoint package. |
| OccamLGS | `/root/baselines/OccamLGS` | `eb98bcb` | Public repo cloned. The nested `diff-gaussian-rasterization/third_party/glm` TLS failure was resolved by a targeted submodule retry; LERF compatibility `language_features` are now complete. The stale `simple_knn` ABI blocker was cleared by rebuilding it into `output/baselines/occamlgs/local_site`; all four LERF compatibility scenes completed RGB training, language-feature extraction for levels 1/2/3, test feature-map rendering, and a normalized pre-rendered metric readout with object-weighted LocAcc 0.8221 / mIoU 0.4515 over 208 objects. Provenance caveat remains: this is a compatibility readout, not a strict released-checkpoint OccamLGS macro. |
| GAGS | `/root/baselines/GAGS` | `4ce2721` | Public repo cloned with submodules. README says training/evaluation code and GT labels are released, but pretrained models/preprocessed datasets are still not released. The vendored `simple_knn` and `segment-anything` packages are installed into `output/baselines/gags/local_site`, and `train.py --help` plus `render.py --help` reach CLI help under that local site. |
| Dr. Splat | `/root/baselines/Dr-Splat` | `764f608` | Public repo cloned. README gives preprocessing/training/render activation commands, but evaluation is marked TBA, so a local wrapper is needed for fair T2/T3 tables. The vendored `simple_knn`, `langsplat-rasterization`, and `segment-anything` packages are installed into `output/baselines/dr_splat/local_site`, and `train.py --help` plus `render_activation.py --help` reach CLI help under that local site. |
| LangSplat | `/root/baselines/LangSplat` | `d70edb8` | Public repo cloned with submodules. The vendored `simple_knn`, `langsplat-rasterization`, and `segment-anything-langsplat` packages are built into `output/baselines/langsplat/local_site` with NumPy pinned to 1.26.4, and `train.py --help`, `render.py --help`, plus `eval/evaluate_iou_loc.py --help` reach CLI help under that local site. Strict comparison still requires pretrained checkpoints or a fresh same-protocol preprocessing/training/eval pass plus same-evaluator metric export. |
| LEGaussians | `/root/baselines/LEGaussians` | `c2230af` | Public repo cloned; renderer and `simple-knn` submodules are present. A local `.gitmodules` repair initializes `preprocess/segment-anything` at `6fdee8f` from `facebookresearch/segment-anything` and `preprocess/segment-anything-langsplat` at `e5dbe4b` from `minghanqin/segment-anything-langsplat`, so recursive submodule status is clean. The vendored `simple_knn` and `diff_gaussian_rasterization` extensions are built into `output/baselines/legaussians/local_site`, and `train.py --help` reaches CLI help under that local site. Strict comparison still requires dataset-specific feature preprocessing, training/rendering, and same-evaluator metric export. |
| CAGS | `/root/baselines/CAGS` | `9136592` | Public repo cloned. README provides OpenGaussian-compatible LERF training, text-rendering, and IoU scripts. The stale global `ashawkey_diff_gaussian_rasterization` ABI issue was cleared by rebuilding the bundled rasterizer zip into `output/baselines/cags/local_site`; incompatible prebuilt PyG wheels were replaced with local `torch-scatter`/`torch-cluster` source builds, and `hdbscan` was installed into the same local site. `gaussian_renderer`, `train.py --help`, and `render_lerf_by_text.py --help` now import successfully under that local site path. Strict local reproduction still needs data-path setup and same-evaluator metric export. ScanNet evaluation is marked TODO upstream. |
| Semantic Gaussians | `/root/baselines/semantic-gaussians` | `ae53137` | Public repo cloned with submodules. The vendored `simple_knn`, `rgbd-rasterization`, `channel-rasterization`, and `segment-anything` packages are built into `output/baselines/semantic_gaussians/local_site`, with compatible NumPy/scikit-image/viser/TensorFlow imports; `train.py` and `fusion.py` import successfully. Strict eval remains blocked because `eval_segmentation.py` and `distill.py` require MinkowskiEngine, and `MinkowskiEngine==0.5.4` fails to build against the host PyTorch 2.7.1/CUDA headers at `spmm.cu`; `view_viser.py` additionally needs PyTorch-Encoding (`encoding`) for LSeg. |
| LaGa | `/root/baselines/LaGa` | `9df7586` | Public repo cloned. README targets view-dependent semantics through object decomposition and semantic descriptors, with LERF-OVS and ScanNet data entry points. A local `.gitmodules` mapping restores `third_party/kmeans_pytorch`, recursive submodule status is clean, `simple_knn` plus both diff rasterizers are built into `output/baselines/laga/local_site`, and `train_scene.py --help` plus `train_affinity_features.py --help` reach CLI help with a chunked `torch.cdist` fallback replacing the incompatible PyTorch3D KNN import. Strict comparison still needs affinity-feature training plus adaptation of `inference.ipynb` to export same-evaluator masks. |
| the unpublished protocol source | `/root/baselines/the unpublished protocol source` | `-` | Published arXiv context row only. Current arXiv source says the code will be publicly released upon acceptance, and no public implementation was found in arXiv metadata/source or web search on 2026-05-18. Strict same-protocol reproduction is blocked until code or checkpoints are released. |

Public source verification was done against the official GitHub pages:
OpenGaussian, LangSplatV2, OccamLGS, GAGS, Dr. Splat, LangSplat,
LEGaussians, CAGS, Semantic Gaussians, and LaGa, plus the the unpublished protocol source arXiv source.

The local state is now also machine-audited by
`radio_gs/scripts/audit_external_baselines.py`, with generated outputs at
`output/baselines/external_baseline_audit/external_baseline_audit.json` and
`output/baselines/external_baseline_audit/external_baseline_audit.md`. A public
snapshot is copied into `paper/artifacts/`. The audit payload now also records
the tracked OccamLGS all-scene pre-rendered readout JSONs and their
object-weighted macro, plus the P1 repository clone state. Submodule command
failures, such as the LEGaussians missing `.gitmodules` mapping, are treated as
missing-submodule blockers rather than clean checkouts.

The audit helper also provides a guarded compatibility switch:
`--link-complete-langsplat`. It creates `scene/language_features` symlinks only
when every image in the scene has matching `langsplat/language_features/*_s.npy`
and `*_f.npy` files. This prevents partial smoke assets from being silently used
as OpenGaussian/OccamLGS inputs.

## Dataset/Artifact Gaps

OpenGaussian and OccamLGS both expect LERF folders with per-frame
`language_features` assets. The local LERF image/COLMAP/label folders exist,
and the guarded compatibility symlinks now provide complete per-frame feature
pairs for all four scenes. The remaining caveat is provenance: these are
VALA/LangSplat-compatible assets from the patched extraction path, not a strict
official OpenGaussian extraction/rerun.

I started VALA/LangSplat-compatible language-feature generation passes using
the LangSplat SAM fork and the SAM-H checkpoint. A representative resume
command is:

```bash
CUDA_VISIBLE_DEVICES=3 \
RADIO_GS_SITE_PACKAGES=/root/baselines/segment-anything-langsplat:/root/miniconda3/envs/iclpose/lib/python3.9/site-packages \
PYTHONPATH=/root/baselines/VALA:${PYTHONPATH:-} \
bash /root/RADIO-GS/radio_gs/scripts/run_repo_python.sh /root/baselines/VALA/run_sam.py \
  --root_dir /root/baselines/VALA \
  --dataset_name lerf_ovs \
  --rep 3dgs \
  --scene ramen \
  --get_semantic \
  --use_langsplat \
  --skip_mask_nms \
  --sam_checkpoint /root/baselines/VALA/ckpts/sam_vit_h_4b8939.pth
```

These passes validate that the local data path and custom LangSplat SAM package
can write `*_s.npy`/`*_f.npy` assets. They are not yet strict official baseline
numbers because `--skip_mask_nms` is an explicit speed patch to avoid the
CPU-bound NMS bottleneck.

Follow-up debugging found three reproducibility issues in the VALA helper path:

- `create_langsplat` skipped a frame when `instance_file` already existed even
  if semantic `*_s.npy`/`*_f.npy` files were missing. I patched the external
  helper at `/root/baselines/VALA/run_sam.py` so semantic generation skips only
  when the semantic files already exist.
- OpenCLIP can still issue a HuggingFace HEAD request even with cached weights.
  Use `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` for resume runs on this host.
- The helper was extended with `--image_shard_count` and `--image_shard_index`
  so one scene can be split across GPUs by sorted image index without duplicate
  writes.
- A late-frame tail can still exceed short foreground timeouts after the shard
  has skipped many already-complete images. I added `--image_stems` so exact
  missing frames can be resumed directly without replaying a full shard.

After the patch, a 90-second offline smoke entered the first missing semantic
frame rather than skipping all 131 frames. Longer sharded and targeted resume
windows then advanced the LangSplat-format assets to:

| Scene | Images | LangSplat `*_s.npy`/`*_f.npy` pairs | Complete |
|---|---:|---:|---:|
| `figurines` | 299 | 299/299 | true |
| `ramen` | 131 | 131/131 | true |
| `teatime` | 177 | 177/177 | true |
| `waldo_kitchen` | 187 | 187/187 | true |

The audit now treats equal partial counts as incomplete unless every image stem
has both feature files. After the sharded and targeted passes,
`--link-complete-langsplat` created guarded `language_features` symlinks for
`figurines`, `ramen`, `teatime`, and `waldo_kitchen`. These remain compatibility
assets because they use `--skip_mask_nms`, so they are suitable for pipeline
smoke/evaluator bring-up but not yet a strict official OpenGaussian/OccamLGS
baseline package.

Using those guarded compatibility assets, the local OpenGaussian LERF rerun
completed at `/root/RADIO-GS/output/baselines/opengaussian/lerf_compat_20260518`.
The evaluator JSON is
`output/baselines/opengaussian/lerf_compat_20260518/opengaussian_lerf_eval.json`.

| Scene | Objects | Missing predictions | mIoU | Acc@0.25 | Acc@0.5 |
|---|---:|---:|---:|---:|---:|
| `figurines` | 56 | 1 | 0.5505 | 0.7143 | 0.6607 |
| `ramen` | 71 | 11 | 0.2806 | 0.4366 | 0.2394 |
| `teatime` | 59 | 3 | 0.5623 | 0.6949 | 0.6271 |
| `waldo_kitchen` | 22 | 6 | 0.3158 | 0.5000 | 0.3636 |
| macro | 208 | 21 | 0.4273 | 0.5865 | 0.4727 |

Using the same compatibility assets, the local OccamLGS teatime, ramen,
figurines, and waldo_kitchen runs completed RGB checkpoints, three language-feature
checkpoints per scene, and official test-frame feature-map renders:

- RGB checkpoints:
  `output/baselines/occamlgs/lerf_compat_20260518/{teatime,ramen,figurines,waldo_kitchen}/chkpnt30000.pth`
- Language checkpoints:
  `{teatime,ramen,figurines,waldo_kitchen}/chkpnt30000_langfeat_{1,2,3}.pth`
- Test renders:
  `output/baselines/occamlgs/lerf_compat_20260518/{teatime,ramen,figurines,waldo_kitchen}/test/ours_30000_langfeat_{1,2,3}/renders_npy/`

A same-label, same-OpenCLIP local readout over the normalized pre-rendered
Occam feature maps completed on all four LERF scenes using
`radio_gs/scripts/eval_prerendered_lerf_features.py`. The JSON files are
`output/baselines/occamlgs/lerf_compat_20260518/occamlgs_{teatime,ramen,figurines,waldo_kitchen}_lerf_prerendered_eval_script.json`.
This is an all-scene compatibility readout, not an official released-checkpoint
OccamLGS macro.

| Method | Scene | Frames | Objects | LocAcc | mIoU |
|---|---|---:|---:|---:|---:|
| OccamLGS normalized pre-rendered compatibility readout | `figurines` | 4 | 56 | 0.7857 | 0.3385 |
| OccamLGS normalized pre-rendered compatibility readout | `teatime` | 6 | 59 | 0.9322 | 0.4973 |
| OccamLGS normalized pre-rendered compatibility readout | `ramen` | 7 | 71 | 0.7465 | 0.5257 |
| OccamLGS normalized pre-rendered compatibility readout | `waldo_kitchen` | 5 | 22 | 0.8636 | 0.3766 |
| object-weighted all-scene readout | `figurines+ramen+teatime+waldo_kitchen` | 22 | 208 | 0.8221 | 0.4515 |

Using the Occam RGB checkpoint as the 3DGS start point, the local LangSplatV2
compatibility rerun completed all four LERF scenes through all three feature
levels and the official `eval_lerf.py --quick_render` path. The summary is
snapshotted at `paper/artifacts/langsplatv2_lerf_summary.{json,md}`.

| Method | Scene | Queries | LocAcc | mIoU |
|---|---|---:|---:|---:|
| LangSplatV2 local compatibility rerun | `figurines` | 56 | 0.8214 | 0.5965 |
| LangSplatV2 local compatibility rerun | `ramen` | 71 | 0.7183 | 0.5913 |
| LangSplatV2 local compatibility rerun | `teatime` | 59 | 0.2034 | 0.0968 |
| LangSplatV2 local compatibility rerun | `waldo_kitchen` | 22 | 0.7273 | 0.5558 |

`radio_gs/scripts/summarize_langsplatv2_lerf_baseline.py` now parses the
official `eval_lerf.py` logs into
`output/baselines/langsplatv2/lerf_compat_20260518/langsplatv2_lerf_summary.{json,md}`.
The current summary is also snapshotted at
`paper/artifacts/langsplatv2_lerf_summary.{json,md}` with checksums. The
all-scene summary reports scene-mean LocAcc 0.6176 / mIoU 0.4601 and
object-weighted LocAcc 0.6010 / mIoU 0.4487 over 208 queries.

The ramen eval initially failed with `IndexError: list index out of range`
because upstream `eval_lerf.py` indexed `scene.getTrainCameras()` by original
LERF frame IDs even when `--eval` removes every eighth frame into the test
split. I patched `/root/baselines/LangSplatV2/eval_lerf.py` to select cameras
from train+test views by `image_name`, verified by
`output/baselines/langsplatv2/lerf_compat_20260518/test_eval_lerf_view_selection.py`.

CAGS local compatibility training/render/eval now completes on all four LERF
scenes from local OpenGaussian 30k starts. Fixes applied along the way include
the vectorized HDBSCAN label-center reducer
(`test_vectorized_hdbscan_centers.py`), trusted checkpoint-loading wrapper,
CPU-only FAISS fallback (`test_cags_faiss_cpu_fallback.py`), and train/test
render wrapper fix (`test_cags_render_eval_wrapper.py`). The paper snapshot is
`paper/artifacts/cags_lerf_summary.{json,md}`. Scene-mean performance is mIoU
0.2627 / Acc@0.25 0.3997, and object-weighted performance is mIoU 0.2394 /
Acc@0.25 0.3558 over 208 objects. This row is diagnostic and low-scoring, with
34 missing rendered masks counted in the source JSONs.

Classic LangSplat now has complete fp32 dim-3 language features and
split-correct train/test feature renders for all four scenes. The chunked
decoder and split-aware feature-path fixes in
`/root/baselines/LangSplat/eval/evaluate_iou_loc.py` are covered by
`output/baselines/langsplat/lerf_compat_20260518/test_evaluate_iou_loc_chunked_decode.py`.
The paper snapshot is `paper/artifacts/langsplat_classic_lerf_summary.{json,md}`.
The corrected four-scene summary is scene-mean LocAcc 0.7335 / mIoU 0.4433
and object-weighted LocAcc 0.7356 / mIoU 0.4613 over 208 queries.

## Protocol Fixes Applied

- Promoted T2 VPR direct-3D selector policy is now fixed `thr0p25`, not
  `mean+2.5std`.
- `paper/lerf_direct_3d_context_table.tex` now reports the promoted VPR row as
  48.01 mIoU / 67.60 Acc@0.25.
- `paper/artifacts/final_rows.yaml` is the paper-facing registry for T1/T2/T3
  rows and external reproduction status.
- `radio_gs/scripts/validate_final_rows_registry.py` now checks the registry
  against source artifacts for the contextual ScanNet support row and the
  the unpublished protocol source no-code blocker.
- `radio_gs/scripts/validate_paper_claims.py` now guards the paper-facing
  VPR rows against `mean+2.5std` selector promotion and flags unqualified
  ScanNet/global-SOTA leaderboard language.
- `paper/artifacts/` now contains a public snapshot of the frozen evidence
  files and checksums, avoiding reliance on the private `output` symlink.
- `paper/artifacts/lerf_direct_3d_query_audit.md` now snapshots the
  `thr0p25` VPR query-level bootstrap and worst-query audit, so the direct-3D
  failure evidence is available without following the private `output` symlink.
- `paper/artifacts/` now also snapshots the LERF rendered failure analysis,
  Waldo direct-3D failure stratification, and VPR confidence/coverage reports
  plus their machine-readable JSON where available, so the key failure evidence
  cited in the submission status is public and checksummed.
- The paper main table, result audit, direct-3D selection report, published
  context report, VPR protocol card, controlled-baseline gap audit,
  efficiency/cost table, and storage-footprint report are also snapshotted in
  `paper/artifacts/` so the key protocol and main-result evidence is visible
  without dereferencing the private `output` symlink.
- The four per-scene direct-3D result JSON files for the `thr0p25` VPR run and
  the silhouette sweep JSON are snapshotted as public machine-readable support
  for the direct-3D table.
- The RADIO-GS ScanNet v67 direct point-query source JSON and contextual kNN
  support JSON are also snapshotted, matching the OpenGaussian ScanNet public
  snapshot already present in `paper/artifacts/`.
- `paper/artifacts/README.md` now indexes the snapshot by T1/T2/T3 track,
  baseline reproduction status, paper tables, and validation guards.
- Mechanism, ablation, and code-audit reports are now snapshotted too:
  alpha/depth boundary alignment, boundary-error readout, feature-error/text
  relevance, compression/downstream correlation, component ablation,
  train-feature-field audit, and the alpha/depth case-figure manifest.
- The remaining small freeze/readiness, ScanNet diagnostic, rendered-grounding
  diagnostic, VPR diagnostic, SAM3 diagnostic, qualitative-manifest, seed
  robustness, and baseline-source verification reports are also public
  snapshots with checksum entries.
- `paper/artifacts/active_goal_completion_audit.md` now records the
  prompt-to-artifact checklist, in-flight LangSplatV2 rows, and strict-SOTA
  blockers that prevent marking the active goal complete.
- `pytest.ini` now pins repository test discovery to `tests/` and excludes
  generated/output directories, so broad `pytest -q` no longer recurses into
  baseline `local_site` package installs under the `output` symlink.

## External Compatibility Patches

The following external-repo patches were required for local reproduction on the
current PyTorch/CUDA host:

- `/root/baselines/OpenGaussian/gaussian_renderer/__init__.py`: tolerate a
  missing/incompatible PyTorch3D wheel by falling back to squared `torch.cdist`
  KNN distances. The fallback is chunked via `OPEN_GAUSSIAN_KNN_CHUNK` to avoid
  materializing full all-pairs matrices during LeRF post-processing.
- `/root/baselines/LangSplatV2/utils/vq_utils.py`: construct temporary RVQ
  tensors in float32 so fp16 compatibility language features and float64
  MiniBatchKMeans centers do not trip `torch.cdist` dtype checks; create
  quick-render top-k indices on the logits device so CUDA boolean indexing does
  not fail during `eval_lerf.py`.
- `/root/baselines/LangSplatV2/utils/loss_utils.py`: compute cosine loss in
  float32 with `eps=1e-6`, avoiding fp16 zero-vector NaNs on masked pixels.
- OccamLGS feature extraction/rendering must prepend
  `/root/baselines/OccamLGS/submodules/gsplat` inside `RADIO_GS_SITE_PACKAGES`;
  otherwise the shared conda `gsplat` shadows the vendored fork and omits the
  `info["activated"]` field used by `gaussian_feature_extractor.py`.
- `/root/baselines/LaGa/.gitmodules`: restore the missing
  `third_party/kmeans_pytorch` mapping to the pinned
  `subhadarship/kmeans_pytorch` commit. LaGa also needs a local
  `scene/gaussian_model_ff.py` fallback that uses chunked `torch.cdist` when
  PyTorch3D KNN cannot import on this host.
- `/root/baselines/LEGaussians/.gitmodules`: restore the missing preprocess
  mappings for `segment-anything` and `segment-anything-langsplat` at the
  gitlink-pinned commits, using `facebookresearch/segment-anything` and
  `minghanqin/segment-anything-langsplat`.
- LEGaussians native extensions: build vendored `simple-knn` and
  `diff-gaussian-rasterization` into
  `output/baselines/legaussians/local_site` so repo imports do not pick up the
  stale shared conda `simple_knn` ABI.
- GAGS native/local imports: build vendored `simple-knn` and install the
  vendored SAM package into `output/baselines/gags/local_site` to avoid the
  stale shared conda `simple_knn` ABI.
- Dr. Splat native/local imports: build vendored `simple-knn`,
  `langsplat-rasterization`, and the vendored SAM package into
  `output/baselines/dr_splat/local_site` to avoid stale shared ABI imports.
- LangSplat native/local imports: build vendored `simple-knn`,
  `langsplat-rasterization`, and `segment-anything-langsplat` into
  `output/baselines/langsplat/local_site`, then pin NumPy to 1.26.4 because
  the initially pulled NumPy 2.0 target install could not load its bundled
  OpenBLAS library. With the isolated site, `train.py`, `render.py`, and
  `eval/evaluate_iou_loc.py` reach CLI help.
- Semantic Gaussians native/local imports: build vendored `simple-knn`,
  `rgbd-rasterization`, `channel-rasterization`, and the vendored SAM package
  into `output/baselines/semantic_gaussians/local_site`; pin a compatible
  NumPy/scikit-image/viser/TensorFlow import stack so `train.py` and `fusion.py`
  import. Full eval/distillation remains blocked because `MinkowskiEngine==0.5.4`
  fails to compile against the host PyTorch 2.7.1/CUDA headers at `spmm.cu`
  even after the stdlib-distutils and OpenBLAS setup, and the visualizer path
  still needs PyTorch-Encoding (`encoding`) for LSeg.
- OccamLGS and LangSplatV2 checkpoint loads use
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` because the locally generated checkpoints
  include optimizer/numpy scalar state and PyTorch 2.6+ defaults to
  `weights_only=True`.

## Next Benchmark Steps

1. Finish or replace the LERF `language_features` generation with a strict,
   documented extraction policy.
2. Keep the tracked OccamLGS all-scene compatibility readout caveated until a
   strict released-checkpoint/extraction-policy macro is available.
3. Finish the LangSplatV2 all-scene local row by letting `waldo_kitchen`
   complete training and `eval_lerf.py`; integrate it only after the summary
   parser sees the completed log.
4. For Dr. Splat, implement a local query-select-render evaluator wrapper
   because the official repo does not ship one yet.
5. For Semantic Gaussians, use a PyTorch/MinkowskiEngine-compatible environment
   or patch the sparse-convolution dependency before attempting ScanNet export
   under the local 19/15/10 evaluator.
6. Re-check the unpublished protocol source code availability before any strict SOTA comparison; as of
   2026-05-18 it is a no-code arXiv context row, not a reproducible baseline.
7. Do not promote global SOTA language until the P0 methods have local rows
   under the same T1/T2/T3 protocols.
