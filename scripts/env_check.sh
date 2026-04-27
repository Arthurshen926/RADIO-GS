#!/bin/bash
# RADIO-GS Environment Verification Script
# Usage: bash scripts/env_check.sh
# Run this on the cloud server after initial setup.

set -e

PASS=0
FAIL=0

pass() { PASS=$((PASS+1)); echo "  ✅ $1"; }
fail() { FAIL=$((FAIL+1)); echo "  ❌ $1"; }

echo "=========================================="
echo "  RADIO-GS Environment Check"
echo "  $(date)"
echo "=========================================="
echo ""

# ---- Python ----
echo "--- Python ---"
python --version 2>&1 || { fail "Python not found"; exit 1; }
python -c "import sys; sys.exit(0 if sys.implementation.name == 'cpython' else 1)" 2>/dev/null \
  && pass "CPython (good)" \
  || fail "Not CPython! May be PyPy - CUDA will not work."

# ---- PyTorch + CUDA ----
echo "--- PyTorch + CUDA ---"
python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null \
  && pass "PyTorch loaded" \
  || fail "PyTorch import failed"
python -c "
import torch
if torch.cuda.is_available():
    print(f'  CUDA {torch.version.cuda}')
    print(f'  GPU {torch.cuda.get_device_name(0)}')
    print(f'  Devices: {torch.cuda.device_count()}')
    x = torch.randn(4, 1280, 60, 80).cuda()
    print(f'  Tensor test: {x.shape} ✅')
" 2>/dev/null && pass "CUDA + GPU available" || fail "CUDA unavailable"

# ---- gsplat ----
echo "--- gsplat ---"
python -c "
import gsplat; print(f'  gsplat {gsplat.__version__}')
" 2>/dev/null && pass "gsplat OK" || fail "gsplat not installed"

# ---- Key packages ----
echo "--- Dependencies ---"
for pkg in numpy scipy PyYAML tqdm pillow opencv_python matplotlib tensorboard plyfile timm; do
  python -c "import ${pkg%%=*}" 2>/dev/null && pass "$pkg" || fail "$pkg missing"
done

# ---- Project imports ----
echo "--- RADIO-GS imports ---"
IMPORTS=(
  "radio_gs.config:load_config"
  "radio_gs.models.hcd_codec:HCDCodec"
  "radio_gs.models.hybrid_gaussian:HybridFeatureGaussian"
  "radio_gs.models.screen_refiner:ScreenSpaceRefiner"
  "radio_gs.rendering.feature_renderer:FeatureFieldRenderer"
  "radio_gs.heads.depth_head:DepthHead"
  "radio_gs.heads.segmentation_head:SegmentationHead"
  "radio_gs.heads.grounding_head:GroundingHead"
  "radio_gs.losses.distillation_loss:DistillationLoss"
  "radio_gs.models.siglip_projection:SigLIP2FeatureProjection"
)
for entry in "${IMPORTS[@]}"; do
  mod="${entry%%:*}"
  cls="${entry##*:}"
  python -c "from $mod import $cls; print(f'  {mod}:{cls} OK')" 2>/dev/null \
    && pass "$entry" || fail "$entry failed"
done

# ---- Data paths ----
echo "--- Data paths ---"
RADIO_REPO="${RADIO_REPO:-}"
if [ -z "$RADIO_REPO" ]; then
  fail "RADIO_REPO not set (torch.hub will auto-download)"
else
  [ -d "$RADIO_REPO" ] && pass "RADIO_REPO=$RADIO_REPO" || fail "RADIO_REPO path invalid: $RADIO_REPO"
fi

for dir in /data/dataset /data/lerf_ovs /data/3dgs_models /data/output; do
  if [ -d "$dir" ]; then
    used=$(du -sh "$dir" 2>/dev/null | cut -f1)
    pass "$dir ($used)"
  else
    fail "$dir not found"
  fi
done

# ---- Checkpoints ----
echo "--- Checkpoints ---"
for ckpt in checkpoints/siglip2_feat_projection.pth checkpoints/siglip2_text_embeddings_v2.pt; do
  [ -f "$ckpt" ] && pass "$ckpt" || fail "$ckpt missing"
done

# ---- Summary ----
echo ""
echo "=========================================="
echo "  Results: $PASS passed, $FAIL failed"
echo "=========================================="

if [ $FAIL -gt 0 ]; then
  echo "  Some checks failed. See above for details."
  exit 1
else
  echo "  All checks passed! Environment is ready."
  exit 0
fi
