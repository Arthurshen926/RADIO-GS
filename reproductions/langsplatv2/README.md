# LangSplatV2 LERF-2D exact-camera reproduction

This package pins the audited LangSplatV2 release and carries only the camera
resolution correction needed for a valid LERF-2D evaluation.

## Protocol correction

Upstream converts the annotation filename to its original frame index, then
uses that index into `scene.getTrainCameras()`. With `eval=True`, the LLFF
split removes every eighth camera from the train list, so the resulting list
index is not the original frame index. The selected pose can therefore belong
to a different RGB frame.

The patch adds `_select_view_for_label_image` and resolves each annotation by
its exact image stem over `train cameras ∪ test cameras`. This preserves the
released LERF behavior: annotated frames may have either role. A test-only
restriction would be a separate diagnostic protocol, not the paper profile.
The launcher also parses all three `cfg_args` files without executing them and
requires the same source scene, `eval=True`, the expected feature level, and an
otherwise identical checkpoint cohort.

This code-intent reproduction is not a pure held-out-view protocol. In the
current annotations, 15 of 22 labeled frames resolve to the LLFF train split
and 7 to the test split. That differs from OccamLGS's paper wording that test
views are withheld, so comparisons must state the released mixed-role camera
semantics explicitly.

The patch changes both `evaluate` and `evaluate_quick`. It does not alter
feature training, codebooks, similarity readout, thresholding, smoothing, or
aggregation.

## Pinned checkout and minimal patch

```bash
git clone https://github.com/ZhaoYujie2002/LangSplatV2.git \
  /root/baselines/LangSplatV2-clean
git -C /root/baselines/LangSplatV2-clean checkout \
  1667303d5c111a5b62f69b9b8991d80045e92b5f
git -C /root/baselines/LangSplatV2-clean apply \
  /root/RADIO-GS/reproductions/langsplatv2/patches/0001-exact-label-camera-resolution.patch
```

The packaged patch SHA-256 is
`a0ba52f843fdc21a0135f71b2ebe2edb5112c4ef48235a080c3c8828a5b285f3`.
The launcher fails closed unless the checkout is at the pinned commit and its
only tracked diff is exactly this patch.

## Launch

Preflight one scene without using the GPU:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python3.9 \
  reproductions/langsplatv2/run_lerf2d_exact_camera.py \
  --scene teatime \
  --upstream /root/baselines/LangSplatV2-clean \
  --dry-run
```

Run all four scenes sequentially. Each invocation serializes GPU 0 with
`/tmp/radio-gs-gpu0.lock`.

```bash
for scene in figurines teatime ramen waldo_kitchen; do
  /root/miniconda3/envs/cybersim_agent/bin/python3.9 \
    reproductions/langsplatv2/run_lerf2d_exact_camera.py \
    --scene "$scene" \
    --upstream /root/baselines/LangSplatV2-clean
done
```

The default checkpoint root is
`output/baselines/langsplatv2/lerf_compat_20260518`, and outputs are isolated
under
`output/protocol_audit_20260731/langsplatv2_lerf2d_view_fix`.

After all four logs are complete, generate per-frame camera-role manifests and
both requested aggregations:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python3.9 \
  -m radio_gs.scripts.summarize_langsplatv2_lerf_audit
```

The summarizer emits:

- `camera_manifests/<scene>.json`, including exact camera name, sorted camera
  index, LLFF role, and query count for every annotated frame;
- `cohort_summary.json`, including the four scene rows, a scene-equal macro,
  and a 208-query micro.

Upstream logs mIoU only to four decimal places. The cohort's query-micro mIoU
is therefore reconstructed from rounded scene means and is marked as such.
Localization hit counts are recovered exactly.

## Audit-checkout provenance

The July 31 audit was executed from `/root/baselines/LangSplatV2` at the same
pinned commit. That checkout already contained optional visualization plumbing
and local torch/runtime compatibility edits. Its actual `eval_lerf.py` diff
SHA-256 was
`c65abd0c79f06ecc56df6e4a8a5093c8203ba785ebd3ff898cb39246221b8994`;
the full tracked diff SHA-256 was
`31f14de37bb17650526b68f9dc6bd0a904c9322e7be1db2186557e217bbee608`.
Those unrelated edits are recorded in `upstream.lock.json` for provenance but
are deliberately excluded from the reusable patch.

To continue the already-started July 31 cohort from that exact recorded
checkout, use the explicit provenance gate:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python3.9 \
  reproductions/langsplatv2/run_lerf2d_exact_camera.py \
  --scene figurines \
  --upstream /root/baselines/LangSplatV2 \
  --allow-recorded-audit-checkout
```

This flag accepts only the recorded `eval_lerf.py` and full tracked-diff
hashes. It does not broaden the reusable patch or accept arbitrary dirty
files.

For I/O-heavy checkpoints, copy only each level's `cfg_args` and
`chkpnt10000.pth` into an isolated staging tree, keep the checkpoint files
read-only, and pass both roots:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python3.9 \
  reproductions/langsplatv2/run_lerf2d_exact_camera.py \
  --scene figurines \
  --upstream /root/baselines/LangSplatV2 \
  --allow-recorded-audit-checkout \
  --checkpoint-root \
    output/protocol_audit_20260731/runtime/langsplatv2_ckpt_stage \
  --checkpoint-source-root \
    output/baselines/langsplatv2/lerf_compat_20260518
```

The launcher hashes all three source and staged checkpoints, requires
source/target equality, and writes paths, sizes, and both hashes into the scene
launcher manifest before it requests the GPU lock.
