"""Pretrain an oracle segmentation head on GT RADIO features + semantic labels.

This script trains a SegmentationHead on ground-truth 1280d RADIO features
paired with semantic labels, producing a frozen head that can later supervise
feature field training without using semantic labels during GS training.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from radio_gs.data.benchmark_paths import resolve_dataset_type, resolve_semantics_path
from radio_gs.heads.segmentation_head import SegmentationHead, compute_miou, compute_pixel_accuracy


class OracleSegDataset(Dataset):
    """Dataset of (feature_map, semantics) pairs for oracle seg-head training."""

    def __init__(
        self,
        feature_dir: str,
        semantics_dir: str,
        feature_size: Tuple[int, int] = (60, 80),
        dataset_type: str = "replica",
        ignore_index: int = 255,
        min_labeled_ratio: float = 0.01,
    ):
        self.feature_dir = Path(feature_dir)
        self.semantics_dir = Path(semantics_dir)
        self.feature_size = feature_size
        self.dataset_type = resolve_dataset_type(dataset_type)
        self.ignore_index = ignore_index
        self.min_labeled_ratio = min_labeled_ratio

        feat_search_dir = self.feature_dir / "backbone"
        if not feat_search_dir.is_dir():
            feat_search_dir = self.feature_dir

        self.pairs: List[Tuple[Path, Path]] = []
        feat_files = sorted(feat_search_dir.glob("rgb_*.pt"))
        for feat_path in feat_files:
            idx = self._extract_index(feat_path.stem)
            if idx is None:
                continue
            sem_path = resolve_semantics_path(self.semantics_dir, idx, self.dataset_type)
            if sem_path is None or not sem_path.exists():
                continue
            self.pairs.append((feat_path, sem_path))

        print(
            f"OracleSegDataset: {len(self.pairs)} valid pairs from "
            f"{self.feature_dir} + {self.semantics_dir}"
        )

        self._preloaded = False
        self._features: List[torch.Tensor] = []
        self._labels: List[torch.Tensor] = []

    @staticmethod
    def _extract_index(stem: str) -> Optional[int]:
        import re

        match = re.search(r"(\d+)$", stem)
        if match:
            return int(match.group(1))
        return None

    def preload(self):
        print(f"  Preloading {len(self.pairs)} samples...")
        kept = 0
        for i, (feat_path, sem_path) in enumerate(self.pairs):
            sample = self._load_sample(feat_path, sem_path)
            if sample is None:
                continue
            self._features.append(sample["features"].half())
            self._labels.append(sample["labels"])
            kept += 1
            if (i + 1) % 100 == 0:
                print(f"    loaded {i + 1}/{len(self.pairs)}")
        self._preloaded = True
        print(f"  Preload complete. Kept {kept} samples.")

    def _load_sample(self, feat_path: Path, sem_path: Path) -> Optional[Dict[str, torch.Tensor]]:
        feat = torch.load(feat_path, map_location="cpu")
        if feat.dim() == 3:
            feat = feat.unsqueeze(0)
        feat = F.interpolate(
            feat.float(), size=self.feature_size, mode="bilinear", align_corners=False
        ).squeeze(0)

        sem = cv2.imread(str(sem_path), cv2.IMREAD_UNCHANGED)
        if sem is None:
            return None
        sem_t = torch.from_numpy(sem.astype(np.int64))
        sem_t = F.interpolate(
            sem_t.float().unsqueeze(0).unsqueeze(0),
            size=self.feature_size,
            mode="nearest",
        ).squeeze(0).squeeze(0).long()

        labeled = (sem_t != self.ignore_index).float().mean().item()
        if labeled < self.min_labeled_ratio:
            return None

        return {"features": feat, "labels": sem_t}

    def __len__(self) -> int:
        if self._preloaded:
            return len(self._features)
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._preloaded:
            return {
                "features": self._features[idx].float(),
                "labels": self._labels[idx],
            }

        feat_path, sem_path = self.pairs[idx]
        sample = self._load_sample(feat_path, sem_path)
        if sample is None:
            raise RuntimeError(f"Invalid sample unexpectedly reached: {feat_path}")
        return sample


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
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "config": config,
                "training_args": {k: str(v) for k, v in training_args.items()},
                "elapsed_seconds": elapsed_seconds,
            },
            f,
            indent=2,
        )


@torch.no_grad()
def evaluate_oracle_seg(
    head: SegmentationHead,
    loader: DataLoader,
    num_classes: int,
    device: str,
    ignore_index: int,
) -> Dict[str, float]:
    head.eval()
    losses, mious, accs = [], [], []
    for batch in loader:
        feat = batch["features"].to(device)
        labels = batch["labels"].to(device)
        logits = head(feat)
        loss = F.cross_entropy(logits, labels, ignore_index=ignore_index)
        losses.append(loss.item())
        pred = logits.argmax(dim=1)
        mious.append(compute_miou(pred, labels, num_classes=num_classes, ignore_index=ignore_index))
        accs.append(compute_pixel_accuracy(pred, labels, ignore_index=ignore_index))
    return {
        "loss": float(np.mean(losses)) if losses else float("inf"),
        "miou": float(np.mean(mious)) if mious else 0.0,
        "pixel_acc": float(np.mean(accs)) if accs else 0.0,
    }


def train_oracle_seg_head(
    dataset: OracleSegDataset,
    head: SegmentationHead,
    epochs: int,
    lr: float,
    batch_size: int,
    num_classes: int,
    device: str,
    ignore_index: int,
    val_ratio: float,
    val_dataset: Optional[OracleSegDataset],
    validate_every: int,
    log_every: int,
    save_best_path: Optional[str],
    checkpoint_config: Optional[Dict[str, object]],
    training_args: Optional[Dict[str, object]],
) -> Dict[str, float]:
    if hasattr(dataset, "preload") and not dataset._preloaded:
        dataset.preload()
    if val_dataset is not None and hasattr(val_dataset, "preload") and not val_dataset._preloaded:
        val_dataset.preload()

    head = head.to(device)
    if val_dataset is None:
        n = len(dataset)
        n_val = max(1, int(n * val_ratio))
        n_train = max(1, n - n_val)
        indices = torch.randperm(n).tolist()
        train_subset = torch.utils.data.Subset(dataset, indices[:n_train])
        val_subset = torch.utils.data.Subset(dataset, indices[n_train:])
    else:
        train_subset = dataset
        val_subset = val_dataset

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("inf")
    best_state = None
    best_metrics: Dict[str, float] = {}
    best_epoch = 0

    t0 = time.time()
    print(
        f"  Training {epochs} epochs, {len(train_loader)} batches/epoch "
        f"(train={len(train_subset)}, val={len(val_subset)}) ...",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        head.train()
        train_losses = []
        for batch in train_loader:
            feat = batch["features"].to(device)
            labels = batch["labels"].to(device)
            logits = head(feat)
            loss = F.cross_entropy(logits, labels, ignore_index=ignore_index)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        should_validate = epoch == 1 or epoch == epochs or (validate_every > 0 and epoch % validate_every == 0)
        should_log = should_validate or epoch == 1 or epoch == epochs or (log_every > 0 and epoch % log_every == 0)

        if should_validate or should_log:
            val_metrics = evaluate_oracle_seg(
                head=head,
                loader=val_loader,
                num_classes=num_classes,
                device=device,
                ignore_index=ignore_index,
            )
            val_loss = val_metrics["loss"]
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
                best_metrics = val_metrics.copy()
                if save_best_path is not None:
                    save_head_checkpoint(
                        output_path=save_best_path,
                        state_dict=best_state,
                        config=checkpoint_config or {},
                        metrics={**best_metrics, "best_epoch": best_epoch, "best_val_loss": best_val},
                        training_args=training_args or {},
                        elapsed_seconds=time.time() - t0,
                    )

            if should_log:
                avg_train = float(np.mean(train_losses)) if train_losses else float("inf")
                print(
                    f"  E{epoch:04d}  train={avg_train:.5f}  val={val_loss:.5f}  "
                    f"mIoU={val_metrics['miou']:.4f}  pixAcc={val_metrics['pixel_acc']:.4f}  "
                    f"[{time.time() - t0:.0f}s]",
                    flush=True,
                )

    if best_state is not None:
        head.load_state_dict(best_state)
    if best_metrics:
        best_metrics["best_epoch"] = best_epoch
        best_metrics["best_val_loss"] = best_val
    if best_state is not None and save_best_path is not None:
        save_head_checkpoint(
            output_path=save_best_path,
            state_dict=best_state,
            config=checkpoint_config or {},
            metrics=best_metrics,
            training_args=training_args or {},
            elapsed_seconds=time.time() - t0,
        )
    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Pretrain oracle segmentation head")
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--semantics_dir", type=str, required=True)
    parser.add_argument("--val_feature_dir", type=str, default=None)
    parser.add_argument("--val_semantics_dir", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, default="replica", choices=["replica", "scannet", "lerf"])
    parser.add_argument("--head_type", type=str, default="mlp", choices=["linear", "mlp", "adaptor"])
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--feature_dim", type=int, default=1280)
    parser.add_argument("--num_classes", type=int, default=101)
    parser.add_argument("--feature_size", type=str, default="60,80")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--validate_every", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    fH, fW = [int(x) for x in args.feature_size.split(",")]

    print("=== Oracle Segmentation Head Pretraining ===")
    print(f"  Features:   {args.feature_dir}")
    print(f"  Semantics:  {args.semantics_dir}")
    if args.val_feature_dir and args.val_semantics_dir:
        print(f"  Val feat:   {args.val_feature_dir}")
        print(f"  Val sem:    {args.val_semantics_dir}")
    print(f"  Output:     {args.output_path}")
    print(f"  Dataset:    {args.dataset_type}")
    print(f"  Head:       {args.head_type} (hidden={args.hidden_dim}, layers={args.num_layers})")
    print(f"  Size:       {fH}x{fW}")
    print(f"  Device:     {device}")

    dataset = OracleSegDataset(
        feature_dir=args.feature_dir,
        semantics_dir=args.semantics_dir,
        feature_size=(fH, fW),
        dataset_type=args.dataset_type,
        ignore_index=args.ignore_index,
    )
    if len(dataset) == 0:
        print("ERROR: No valid feature-semantics pairs found!")
        sys.exit(1)

    val_dataset = None
    if args.val_feature_dir or args.val_semantics_dir:
        if not args.val_feature_dir or not args.val_semantics_dir:
            print("ERROR: --val_feature_dir and --val_semantics_dir must be provided together!")
            sys.exit(1)
        val_dataset = OracleSegDataset(
            feature_dir=args.val_feature_dir,
            semantics_dir=args.val_semantics_dir,
            feature_size=(fH, fW),
            dataset_type=args.dataset_type,
            ignore_index=args.ignore_index,
        )
        if len(val_dataset) == 0:
            print("ERROR: No valid validation feature-semantics pairs found!")
            sys.exit(1)

    head = SegmentationHead(
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        head_type=args.head_type,
    )
    print(f"  Head params: {sum(p.numel() for p in head.parameters()) / 1e6:.3f}M")

    metrics = train_oracle_seg_head(
        dataset=dataset,
        val_dataset=val_dataset,
        head=head,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        device=device,
        ignore_index=args.ignore_index,
        val_ratio=0.1,
        validate_every=args.validate_every,
        log_every=args.log_every,
        save_best_path=args.output_path,
        checkpoint_config={
            "head_type": args.head_type,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "feature_dim": args.feature_dim,
            "num_classes": args.num_classes,
        },
        training_args=vars(args),
    )

    print(f"\n=== Oracle Segmentation Head Trained ({metrics.get('best_epoch', 0)} best epoch) ===")
    print(
        f"  mIoU={metrics.get('miou', 0.0):.4f}  "
        f"pixAcc={metrics.get('pixel_acc', 0.0):.4f}"
    )
    print(f"  Saved to: {args.output_path}")
    print(f"  Report:   {args.output_path.replace('.pth', '_report.json')}")


if __name__ == "__main__":
    main()
