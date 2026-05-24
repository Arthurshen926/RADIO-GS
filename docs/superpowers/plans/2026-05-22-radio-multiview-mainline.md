# RADIO Multiview Mainline Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the expert recommendations from `ChatGPT-RADIO模型多视角重建优化 (2).md` until the main paper line has verified positive evidence or a documented negative diagnostic.

**Architecture:** Diagnose implementation risks first, then promote only method changes that improve validated metrics. SAM3 work is split into official state diagnostics, official-decoder bridge upper bounds, and method-internal SAM-style readout. Direct-field work is split into cache alignment, head consistency, one-scene overfit, and deployable primitive readout improvements.

**Tech Stack:** PyTorch, official SAM3 processor, CTF-GS/HCD model checkpoints, VPR feature caches, LERF-OVS direct-3D evaluator, pytest.

---

### Task 1: SAM3 Official Decoder Diagnostics

**Files:**
- Create: `radio_gs/scripts/diagnose_sam3_decoder_state.py`
- Test: `tests/test_sam3_decoder_diagnostics.py`
- Reference: `radio_gs/scripts/build_sam3_foundation_cache.py`
- Reference: `radio_gs/scripts/train_sam3_decoder_bridge.py`

- [ ] **Step 1: Add pure helpers**

Implement helpers for recursive tensor state cloning, tensor summaries, mask IoU, and GT-mask-to-normalized-cxcywh conversion. These helpers must not load SAM3 in tests.

- [ ] **Step 2: Add identity injection mode**

Run `processor.set_image(image)` once, clone the returned state, then call `set_text_prompt(query, cloned_state)`. Compare the resulting mask to the baseline `set_text_prompt(query, original_state)` path.

- [ ] **Step 3: Add GT-box diagnostic mode**

For LERF-labeled frames, build the GT mask box in SAM3's documented normalized `[cx, cy, w, h]` format and call `add_geometric_prompt`.

- [ ] **Step 4: Save JSON evidence**

Write baseline-vs-identity IoU, GT-box IoU, mask areas, state tensor summaries, resolution, dtype, and any exception string.

- [ ] **Step 5: Verify**

Run:

```bash
bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_sam3_decoder_diagnostics.py
bash radio_gs/scripts/run_repo_python.sh -m py_compile radio_gs/scripts/diagnose_sam3_decoder_state.py
```

Expected: tests pass and compile exits 0.

### Task 2: VPR Cache Geometry Alignment

**Files:**
- Create: `radio_gs/scripts/audit_vpr_cache_alignment.py`
- Test: `tests/test_vpr_cache_alignment.py`
- Reference: `radio_gs/scripts/eval_scannet_pointcloud_radio_gs.py`

- [ ] **Step 1: Add xyz statistics helpers**

Compute count match, max/mean/p95 L2 distance, scene scale, normalized max L2, and SHA256 hashes for model xyz and cache xyz.

- [ ] **Step 2: Add CLI**

Load `--config`, `--checkpoint`, and `--teacher_cache`; compare `payload["xyz"]` against `model.get_xyz()`.

- [ ] **Step 3: Add fail policy**

Return non-zero or `status=failed` when `max_l2 > --fail_max_l2`, default `1e-5`.

- [ ] **Step 4: Verify all LERF VPR caches**

Run the audit on figurines, ramen, teatime, and waldo_kitchen base checkpoints.

### Task 3: Direct Head Consistency Diagnostics

**Files:**
- Create: `radio_gs/scripts/diagnose_direct_head_consistency.py`
- Test: `tests/test_direct_head_consistency.py`
- Reference: `radio_gs/scripts/eval_lerf_direct_3d_selection.py`
- Reference: `radio_gs/scripts/train_scannet_point_summary_adapter.py`

- [ ] **Step 1: Add summary cosine and rank helpers**

Compute mean/p10/p50/p90 cosine to VPR summary targets, text-ranking KL/agreement when text embeddings are provided, and an index hash.

- [ ] **Step 2: Add eval-path extraction**

For sampled Gaussian indices, reproduce the same direct summary path used by LERF direct-3D evaluation: decoded+SigLIP2 head, optional point summary adapter, optional blend.

- [ ] **Step 3: Add metadata warnings**

Warn when the checkpoint contains `point_summary_adapter_state_dict` but the diagnosis/eval settings do not enable it.

- [ ] **Step 4: Verify**

Run on the known figurines base, old VPR-field checkpoint, and latest negative checkpoint to explain why the latest prompt/rank run should not be promoted.

### Task 4: Direct Primitive Readout Improvement

**Files:**
- Modify: `radio_gs/scripts/train_scannet_point_summary_adapter.py`
- Modify: `radio_gs/scripts/eval_lerf_direct_3d_selection.py`
- Test: existing direct adapter tests or new focused tests.

- [ ] **Step 1: Add method-only confidence/objectness head if diagnostics justify it**

Learn `rho_i = sigmoid(C(compact_i))` from VPR view-count or score-margin targets and store only global head weights in the checkpoint.

- [ ] **Step 2: Add direct-score gating**

At inference, use `score_i(q) = rho_i * cos(B(compact_i), text_q)` without reading VPR cache.

- [ ] **Step 3: Run one-scene overfit**

Train only the primitive adapter/head on one scene and verify the direct summary reproduces the VPR summary distribution.

- [ ] **Step 4: Run LERF direct-3D smoke**

Evaluate figurines and waldo_kitchen first. Promote only if mIoU/Acc@0.25 improves over the current compact-field reference.

### Task 5: Evidence and Paper Mainline Update

**Files:**
- Create or update: `paper/artifacts/radio_multiview_mainline_audit_20260522.md`
- Update only if positive: `paper/artifacts/final_rows.yaml`
- Update only if positive: `paper/radio_gs_draft.tex`

- [ ] **Step 1: Record diagnostics**

Document each diagnostic with exact commands, output JSON paths, metric deltas, and promotion decision.

- [ ] **Step 2: Keep negative branches out of main tables**

Do not update canonical rows for failed official decoder bridge or failed direct-field variants.

- [ ] **Step 3: Final verification**

Run targeted pytest, py_compile, `validate_final_rows_registry.py`, and `validate_paper_claims.py`.

