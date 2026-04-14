"""Pretrain an oracle depth head on GT RADIO features + GT depth.

This script trains a DepthHead on ground-truth 1280d RADIO features
paired with ground-truth depth maps, producing a frozen "oracle" head
that can later supervise feature field training.

Usage:
    python radio_gs/scripts/pretrain_oracle_head.py \
        --feature_dir /path/to/radio_features_1280d/room_0/train \
        --depth_dir /path/to/room_0/train/depth \
        --output_path output/radio_gs/oracle_heads/room_0_depth_head.pth \
        [--head_type mlp] [--hidden_dim 256] [--num_layers 3] \
        [--epochs 500] [--lr 1e-3] [--batch_size 8] \
        [--feature_size 60,80] [--gpu 0]

For domain-matched training, first export rendered features with
`render_codec_features.py`, then train here on the rendered train split and
optionally validate on an explicit rendered val split:
    python radio_gs/scripts/pretrain_oracle_head.py \
        --feature_dir output/.../train/backbone \
        --depth_dir output/.../train/depth \
        --val_feature_dir output/.../val/backbone \
        --val_depth_dir output/.../val/depth \
        --output_path output/radio_gs/oracle_heads/room_0_dm_depth_head.pth
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from radio_gs.heads.depth_head import DepthHead, DepthLoss


class OracleDepthDataset(Dataset):
    """Dataset of (feature_map, depth_map) pairs for oracle head training."""

    def __init__(
        self,
        feature_dir: str,
        depth_dir: str,
        feature_size: Tuple[int, int] = (60, 80),
        depth_scale: float = 1000.0,
        min_valid_ratio: float = 0.1,
    ):
        self.feature_dir = Path(feature_dir)
        self.depth_dir = Path(depth_dir)
        self.feature_size = feature_size
        self.depth_scale = depth_scale
        self.min_valid_ratio = min_valid_ratio

        # Find matching pairs
        self.pairs: List[Tuple[Path, Path]] = []

        # Support both flat (feat_0.pt) and RADIO backbone dir (backbone/rgb_0.pt)
        feat_search_dir = self.feature_dir
        if (self.feature_dir / "backbone").is_dir():
            feat_search_dir = self.feature_dir / "backbone"

        feat_files = sorted(feat_search_dir.glob("*.pt"))
        for feat_path in feat_files:
            stem = feat_path.stem
            idx = self._extract_index(stem)
            if idx is None:
                continue
            depth_path = self.depth_dir / f"depth_{idx}.png"
            if depth_path.exists():
                self.pairs.append((feat_path, depth_path))

        print(f"OracleDepthDataset: {len(self.pairs)} valid pairs "
              f"from {self.feature_dir} + {self.depth_dir}")

        # Preload all data into memory for fast training
        self._preloaded = False
        self._features: List[torch.Tensor] = []
        self._depths: List[torch.Tensor] = []
        self._masks: List[torch.Tensor] = []

    def preload(self):
        """Load all data into memory for fast training."""
        print(f"  Preloading {len(self.pairs)} samples...")
        for i, (feat_path, depth_path) in enumerate(self.pairs):
            feat = torch.load(feat_path, map_location="cpu")
            if feat.dim() == 3:
                feat = feat.unsqueeze(0)
            feat = F.interpolate(
                feat.float(), size=self.feature_size,
                mode="bilinear", align_corners=False,
            ).squeeze(0)

            depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            depth = torch.from_numpy(depth_img.astype(np.float32) / self.depth_scale)
            depth = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0), size=self.feature_size,
                mode="bilinear", align_corners=False,
            ).squeeze()

            self._features.append(feat.half())  # Store as fp16 to save RAM
            self._depths.append(depth.half())
            self._masks.append(depth > 0.01)

            if (i + 1) % 100 == 0:
                print(f"    loaded {i + 1}/{len(self.pairs)}")

        self._preloaded = True
        print(f"  Preload complete.")

    @staticmethod
    def _extract_index(stem: str) -> Optional[int]:
        """Extract numeric index from feature filename."""
        import re
        match = re.search(r"(\d+)$", stem)
        if match:
            return int(match.group(1))
        return None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._preloaded:
            return {
                "features": self._features[idx].float(),
                "depth": self._depths[idx].float(),
                "valid_mask": self._masks[idx],
            }

        feat_path, depth_path = self.pairs[idx]

        # Load feature
        feat = torch.load(feat_path, map_location="cpu")
        if feat.dim() == 3:
            feat = feat.unsqueeze(0)  # [1, C, H, W]
        feat = F.interpolate(
            feat.float(),
            size=self.feature_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # [C, H, W]

        # Load depth
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        depth = torch.from_numpy(depth_img.astype(np.float32) / self.depth_scale)
        depth = F.interpolate(
            depth.unsqueeze(0).unsqueeze(0),
            size=self.feature_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze()  # [H, W]

        valid = depth > 0.01
        return {
            "features": feat,
            "depth": depth,
            "valid_mask": valid,
        }


def save_head_checkpoint(
    output_path: str,
    state_dict: Dict[str, torch.Tensor],
    config: Dict[str, object],
    metrics: Dict[str, float],
    training_args: Dict[str, object],
    elapsed_seconds: float,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    checkpoint = {
        "state_dict": state_dict,
        "config": config,
        "metrics": metrics,
        "training_args": training_args,
    }
    torch.save(checkpoint, output_path)

    report_path = output_path.replace(".pth", "_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "metrics": metrics,
            "config": config,
            "training_args": {k: str(v) for k, v in training_args.items()},
            "elapsed_seconds": elapsed_seconds,
        }, f, indent=2)


def train_oracle_head(
    dataset: OracleDepthDataset,
    head: DepthHead,
    loss_fn: DepthLoss,
    epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 8,
    device: str = "cuda",
    val_ratio: float = 0.1,
    val_dataset: Optional[OracleDepthDataset] = None,
    validate_every: int = 10,
    log_every: int = 25,
    save_best_path: Optional[str] = None,
    checkpoint_config: Optional[Dict[str, object]] = None,
    training_args: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
    """Train oracle depth head and return final metrics."""
    # Preload data for fast training
    if hasattr(dataset, 'preload') and not dataset._preloaded:
        dataset.preload()
    if val_dataset is not None and hasattr(val_dataset, 'preload') and not val_dataset._preloaded:
        val_dataset.preload()

    head = head.to(device)
    head.train()

    if val_dataset is None:
        # Train/val split from a single dataset
        n = len(dataset)
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_val
        indices = torch.randperm(n).tolist()
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)
        nw_train = 0 if getattr(dataset, '_preloaded', False) else 2
        nw_val = nw_train
    else:
        train_subset = dataset
        val_subset = val_dataset
        nw_train = 0 if getattr(dataset, '_preloaded', False) else 2
        nw_val = 0 if getattr(val_dataset, '_preloaded', False) else 2

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=nw_train, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=nw_val, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    best_metrics = {}
    best_epoch = 0

    import time as _time
    t0 = _time.time()
    print(
        f"  Training {epochs} epochs, {len(train_loader)} batches/epoch "
        f"(train={len(train_subset)}, val={len(val_subset)}) ...",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        # Train
        head.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            feat = batch["features"].to(device)
            depth = batch["depth"].to(device).unsqueeze(1)  # [B, 1, H, W]
            mask = batch["valid_mask"].to(device).unsqueeze(1)

            pred = head(feat)
            loss = loss_fn(pred, depth, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = train_loss / max(n_batches, 1)

        # Validate & log
        should_validate = (
            epoch == 1
            or epoch == epochs
            or (validate_every > 0 and epoch % validate_every == 0)
        )
        should_log = (
            should_validate
            or epoch == epochs
            or epoch == 1
            or (log_every > 0 and epoch % log_every == 0)
        )

        if should_validate or should_log:
            head.eval()
            val_metrics = evaluate_oracle(head, val_loader, device)
            val_loss = val_metrics["loss"]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
                best_metrics = val_metrics.copy()
                best_epoch = epoch
                if save_best_path is not None:
                    metrics_to_save = best_metrics.copy()
                    metrics_to_save["best_epoch"] = best_epoch
                    metrics_to_save["best_val_loss"] = best_val_loss
                    save_head_checkpoint(
                        output_path=save_best_path,
                        state_dict=best_state,
                        config=checkpoint_config or {},
                        metrics=metrics_to_save,
                        training_args=training_args or {},
                        elapsed_seconds=_time.time() - t0,
                    )

            if should_log:
                elapsed = _time.time() - t0
                print(f"  E{epoch:04d}  train={avg_train:.5f}  "
                      f"val={val_loss:.5f}  "
                      f"AbsRel={val_metrics['abs_rel']:.4f}  "
                      f"RMSE={val_metrics['rmse']:.4f}  "
                      f"δ<1.25={val_metrics['delta1']:.4f}  "
                      f"[{elapsed:.0f}s]", flush=True)

    # Restore best
    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()

    if best_metrics:
        best_metrics["best_epoch"] = best_epoch
        best_metrics["best_val_loss"] = best_val_loss
    if best_state is not None and save_best_path is not None:
        save_head_checkpoint(
            output_path=save_best_path,
            state_dict=best_state,
            config=checkpoint_config or {},
            metrics=best_metrics,
            training_args=training_args or {},
            elapsed_seconds=_time.time() - t0,
        )

    return best_metrics


@torch.no_grad()
def evaluate_oracle(
    head: DepthHead,
    loader: DataLoader,
    device: str = "cuda",
) -> Dict[str, float]:
    """Evaluate oracle head on depth metrics."""
    head.eval()
    abs_rels, rmses, delta1s, losses = [], [], [], []
    loss_fn = DepthLoss(loss_type="scale_invariant")

    for batch in loader:
        feat = batch["features"].to(device)
        depth = batch["depth"].to(device).unsqueeze(1)
        mask = batch["valid_mask"].to(device).unsqueeze(1)

        pred = head(feat)
        loss = loss_fn(pred, depth, mask)
        losses.append(loss.item())

        # Per-image metrics
        B = pred.shape[0]
        for b in range(B):
            p = pred[b, 0]
            g = depth[b, 0]
            m = mask[b, 0].bool()
            if m.sum() < 10:
                continue
            pv, gv = p[m], g[m]
            abs_rels.append(((pv - gv).abs() / gv.clamp(min=1e-6)).mean().item())
            rmses.append(((pv - gv) ** 2).mean().sqrt().item())
            delta = torch.max(pv / gv.clamp(min=1e-6), gv / pv.clamp(min=1e-6))
            delta1s.append((delta < 1.25).float().mean().item())

    return {
        "loss": np.mean(losses) if losses else float("inf"),
        "abs_rel": np.mean(abs_rels) if abs_rels else float("inf"),
        "rmse": np.mean(rmses) if rmses else float("inf"),
        "delta1": np.mean(delta1s) if delta1s else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Pretrain oracle depth head")
    parser.add_argument("--feature_dir", type=str, required=True,
                        help="Directory with GT RADIO feature .pt files")
    parser.add_argument("--depth_dir", type=str, required=True,
                        help="Directory with depth_*.png files")
    parser.add_argument("--val_feature_dir", type=str, default=None,
                        help="Optional validation feature directory")
    parser.add_argument("--val_depth_dir", type=str, default=None,
                        help="Optional validation depth directory")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to save the trained head checkpoint")
    parser.add_argument("--head_type", type=str, default="mlp",
                        choices=["linear", "mlp", "dpt"])
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--feature_dim", type=int, default=1280)
    parser.add_argument("--feature_size", type=str, default="60,80",
                        help="Feature map size as 'H,W'")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--validate_every", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--loss_type", type=str, default="berhu",
                        choices=["l1", "scale_invariant", "berhu"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--depth_scale", type=float, default=1000.0,
                        help="Depth PNG to meters scale factor")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    fH, fW = [int(x) for x in args.feature_size.split(",")]

    print(f"=== Oracle Depth Head Pretraining ===")
    print(f"  Features: {args.feature_dir}")
    print(f"  Depth:    {args.depth_dir}")
    if args.val_feature_dir and args.val_depth_dir:
        print(f"  Val feat: {args.val_feature_dir}")
        print(f"  Val dpth: {args.val_depth_dir}")
    print(f"  Output:   {args.output_path}")
    print(f"  Head:     {args.head_type} (hidden={args.hidden_dim}, layers={args.num_layers})")
    print(f"  Size:     {fH}×{fW}")
    print(f"  Device:   {device}")

    # Build dataset
    dataset = OracleDepthDataset(
        feature_dir=args.feature_dir,
        depth_dir=args.depth_dir,
        feature_size=(fH, fW),
        depth_scale=args.depth_scale,
    )
    if len(dataset) == 0:
        print("ERROR: No valid feature-depth pairs found!")
        sys.exit(1)

    val_dataset = None
    if args.val_feature_dir or args.val_depth_dir:
        if not args.val_feature_dir or not args.val_depth_dir:
            print("ERROR: --val_feature_dir and --val_depth_dir must be provided together!")
            sys.exit(1)
        val_dataset = OracleDepthDataset(
            feature_dir=args.val_feature_dir,
            depth_dir=args.val_depth_dir,
            feature_size=(fH, fW),
            depth_scale=args.depth_scale,
        )
        if len(val_dataset) == 0:
            print("ERROR: No valid validation feature-depth pairs found!")
            sys.exit(1)

    # Build head
    head = DepthHead(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        head_type=args.head_type,
    )
    print(f"  Head params: {sum(p.numel() for p in head.parameters()) / 1e6:.3f}M")

    # Loss
    loss_fn = DepthLoss(loss_type=args.loss_type)

    # Train
    t0 = time.time()
    checkpoint_config = {
        "head_type": args.head_type,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "feature_dim": args.feature_dim,
    }
    metrics = train_oracle_head(
        dataset=dataset,
        val_dataset=val_dataset,
        head=head,
        loss_fn=loss_fn,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        validate_every=args.validate_every,
        log_every=args.log_every,
        save_best_path=args.output_path,
        checkpoint_config=checkpoint_config,
        training_args=vars(args),
    )
    elapsed = time.time() - t0

    # Save checkpoint
    checkpoint = torch.load(args.output_path, map_location="cpu")

    print(f"\n=== Oracle Head Trained ({elapsed:.1f}s) ===")
    print(f"  AbsRel={metrics['abs_rel']:.4f}  RMSE={metrics['rmse']:.4f}  "
          f"δ<1.25={metrics['delta1']:.4f}")
    if "best_epoch" in metrics:
        print(f"  Best epoch: E{int(metrics['best_epoch']):04d}")
    print(f"  Saved to: {args.output_path}")

    # Also save a human-readable report
    report_path = args.output_path.replace(".pth", "_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "metrics": metrics,
            "config": checkpoint["config"],
            "training_args": {k: str(v) for k, v in vars(args).items()},
            "elapsed_seconds": elapsed,
        }, f, indent=2)
    print(f"  Report:  {report_path}")


if __name__ == "__main__":
    main()
