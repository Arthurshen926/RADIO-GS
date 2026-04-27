# RADIO-GS Project Status & Goals

> Last updated: 2026-04-25
> Context: Preparing for cloud server migration and systematic paper submission pipeline.

---

## 1. Project Overview

**RADIO-GS** distills frozen RADIO C-RADIOv4-H features (1280d) into a 3D Gaussian Splatting scene representation, enabling novel-view rendering of foundation features that support open-vocabulary grounding, depth estimation, and semantic segmentation.

### Core innovation thesis

> Not "learning task-specific features from scratch", but **reconstructing pre-trained foundation features in 3D space**, with **geometric knowledge distilled back into the feature field via Frozen Depth Head (FDH) supervision** to improve semantic task performance.

Key distinction from prior work:
- LangSplat / LEGaussians: learn task-specific language features → optimize for grounding directly
- RADIO-GS: reconstruct general-purpose foundation features → FDH provides geometric→semantic distillation bridge

---

## 2. Current Status (2026-04-25)

### 2.1 Method Components

| Component | Status | Notes |
|-----------|--------|-------|
| 3DGS geometry backbone | ✅ Complete | Replica (depth-init), LERF (COLMAP), ScanNet pipelines |
| RADIO teacher feature extraction | ✅ Complete | `extract_radio_features.py`, 1280d, frame_manifest |
| Hybrid feature field | ✅ Complete | fine (per-Gaussian latent) + coarse (hash grid) + decoupled fusion |
| HCD codec | ✅ Complete | 1x1 conv, dual-stream, symmetric decoder option, 192d bottleneck |
| Screen-space refiner | ✅ Complete | 8-block residual CNN, 6 guide channels |
| FeatSharp-3D | ✅ Complete | Analytical sharpening mode |
| FDH supervision | ✅ Complete | Frozen depth head, scale_invariant loss, warmup |
| Multi-task training | ✅ Complete | depth + seg + grounding + FDH joint optimization |
| SigLIP2 text grounding | ✅ Complete | Projection + summary head + query aux loss |
| Exact-domain DM head | ✅ Complete | Render → train head → evaluate pipeline |

### 2.2 Evaluation Pipeline

| Component | Status | Notes |
|-----------|--------|-------|
| `eval_rendered.py` (multi-task) | ✅ Production | rendered/fused/geom/direct-head depth + seg + probes |
| `eval_grounding.py` / `eval_lerf_grounding.py` | ✅ Production | SigLIP2 grounding with temperature sweep |
| `generate_visualizations_v2.py` | ✅ Production | PCA/depth/seg/grounding/composite figures |
| Result aggregation scripts | ✅ Production | `build_submission_tables.py`, `aggregate_results.py` |
| Paper figure scripts | ✅ Partially | Basic comparison figures generated |

### 2.3 Experiment Status

**Replica room_0 depth** (best known):
| Metric | Value |
|--------|-------|
| Geom full-res depth | 0.0223 |
| Direct exact-domain DM head | 0.0361 |
| Fused depth | 0.0335 |
| Rendered depth (probe) | 0.0562 |

**LERF-OVS grounding** (best known, draft numbers):
| Scene | Macro LocAcc |
|-------|-------------|
| Figurines | 0.821 |
| Ramen | 0.901 |
| Teatime | 0.881 |
| Waldo Kitchen | 0.864 |
| **Macro avg** | **0.867** |

**Ablation configs** (defined, partial runs completed):
- nofdh_240ep: 5 runs (room_0 + 4 LERF) — **currently running**
- fdh_ws240_240ep: queued after nofdh completion
- pure_frozen / pure_frozen_depth_only: queued

### 2.4 Paper Assets

| Asset | Status |
|-------|--------|
| Paper skeleton (docs/) | ✅ Drafted |
| Submission status analysis | ✅ Documented |
| Benchmarking plan | ✅ Frozen |
| Algorithm framework doc | ✅ Complete (Chinese) |
| Main table draft | ✅ Internal numbers |
| Figures (grounding comparison) | ⚠️ Basic versions done |
| Abstract/Introduction full draft | ❌ Not yet |
| Related Work | ❌ Not yet |
| Method section full draft | ❌ Not yet |

---

## 3. Short-term Goals (Pre-submission, ~6-9 weeks)

### Phase 1 — Complete Current Rerun Queue (Weeks 1-2)
- [ ] Monitor 5x nofdh_240ep to completion
- [ ] Auto-trigger downstream queue (pure_frozen → fdh_ws240 → auto-eval)
- [ ] Verify all FDH / grounding eval results
- [ ] Freeze canonical result tables with provenance

### Phase 2 — Core Missing Experiments (Weeks 2-4)
- [ ] **FDH vs no-FDH systematic ablation** (all 4 LERF scenes + room_0)
- [ ] **ws240 warm-start ablation** (direct FDH vs ws240→FDH)
- [ ] **Refiner on/off ablation**
- [ ] **Bottleneck dimension sweep** (64 vs 128 vs 192)
- [ ] **Feature resolution ablation** (30x40 vs 60x80)
- [ ] **Efficiency table**: training time, GPU memory, inference FPS vs baselines

### Phase 3 — Cross-domain Generalization (Weeks 3-5)
- [ ] **ScanNet** (1-2 scenes): geometry → features → train → eval depth + seg
- [ ] Verify cross-domain feature quality

### Phase 4 — Writing & Publication (Weeks 4-9)
- [ ] Abstract + Introduction full draft
- [ ] Related work survey
- [ ] Method section (hybrid + HCD + refiner + FDH)
- [ ] Experiments section (main results + ablations + cross-domain + efficiency)
- [ ] Publication-quality figures (method overview, qualitative comparisons)
- [ ] Supplementary material

---

## 4. Long-term Goals

### Research Extensions

- **Feature attribution / channel selection analysis**: Use the 1280d feature space + SigLIP2 grounding as a controlled platform to study which feature dimensions encode which semantic concepts, and whether FDH makes feature selection more interpretable.
- **Cross-scene feature transfer**: Analyze whether channel importance patterns learned on one scene transfer to another.
- **Multi-scene joint feature field**: Extend single-scene distillation to a categorical feature field that generalizes across scenes.

### Publication Strategy

| Target | Timeline | Scope |
|--------|----------|-------|
| **ECCV 2026** (Jul deadline) | 6-9 weeks from now | Conference paper: LERF-OVS + Replica + FDH + ablations |
| **TPAMI / IJCV** (extension) | ECCV + 6-12 months | Add ScanNet, efficiency, feature attribution analysis, expanded benchmarks |

---

## 5. Known Technical Debt

### Environment
- Current conda env (iclpose) uses **PyPy 3.9**, which cannot load PyTorch CUDA extensions
- Must use **CPython** (not PyPy) for GPU training
- gsplat 1.4.0 + PyTorch 2.0+ + CUDA (compatible)

### Data
- `output/` is a symlink to NFS mount (`/mnt/pool/sqy/results/RADIO-GS/output/`)
- `RADIO_REPO` environment variable must be set for feature extraction
- Dataset paths are hardcoded in configs; cloud server paths will differ

### Code
- Some configs reference local absolute paths (ply_path, feature_dir)
- RADIO feature extraction is offline (saves .pt files) — needs separate data transfer
- Grounding eval requires SigLIP2 text embedding files (bundled in checkpoints/)
