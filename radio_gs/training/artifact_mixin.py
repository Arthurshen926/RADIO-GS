"""Checkpoint and artifact helpers for RADIO-GS feature-field training."""

from __future__ import annotations

import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from radio_gs.utils.checkpoint_io import load_trusted_checkpoint


class TrainingArtifactMixin:
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> None:
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "codec_state_dict": self.codec.state_dict(),
            "sharpener_state_dict": self.sharpener.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_cosine": self.best_cosine,
            "best_metric_name": self.best_metric_name,
            "best_metric_mode": self.best_metric_mode,
            "best_selection_score": self.best_selection_score,
            "best_selection_value": self.best_selection_value,
            "metrics": metrics,
        }
        if self.use_refiner and self.refiner is not None:
            state["refiner_state_dict"] = self.refiner.state_dict()
        if self.depth_head is not None:
            state["depth_head_state_dict"] = self.depth_head.state_dict()
        if self.seg_head is not None:
            state["seg_head_state_dict"] = self.seg_head.state_dict()
        if self.point_summary_adapter is not None:
            state["point_summary_adapter_state_dict"] = self.point_summary_adapter.state_dict()
            if self.point_summary_adapter_metadata:
                state["point_summary_adapter_metadata"] = dict(
                    self.point_summary_adapter_metadata
                )
            if self.point_summary_adapter_epoch is not None:
                state["point_summary_adapter_epoch"] = int(
                    self.point_summary_adapter_epoch
                )
            if self.point_summary_adapter_best_metric is not None:
                state["point_summary_adapter_best_metric"] = float(
                    self.point_summary_adapter_best_metric
                )
        if self.foundation_cache_projectors:
            state["foundation_cache_projectors_state_dict"] = (
                self.foundation_cache_projectors.state_dict()
            )
        if not getattr(self.cfg, "skip_latest_checkpoint", False):
            torch.save(state, self.ckpt_dir / "latest.pth")
        if is_best:
            torch.save(state, self.ckpt_dir / "best.pth")
            self.best_epoch = epoch
        # Save periodic epoch checkpoint if configured
        periodic = getattr(self.cfg, "save_periodic_every", 0)
        if periodic > 0 and epoch % periodic == 0:
            torch.save(state, self.ckpt_dir / f"epoch_{epoch:03d}.pth")
        self._write_experiment_report(epoch)

    def _config_to_dict(self) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for key, value in vars(self.cfg).items():
            result[key] = str(value) if isinstance(value, Path) else value
        return result

    @staticmethod
    def _safe_jsonify(value):
        if isinstance(value, dict):
            return {str(k): TrainingArtifactMixin._safe_jsonify(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [TrainingArtifactMixin._safe_jsonify(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        return value

    @staticmethod
    def _get_git_metadata() -> Dict[str, object]:
        try:
            import subprocess

            root = Path(__file__).resolve().parents[2]
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "status", "--short"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            return {
                "commit": commit,
                "dirty": bool(status),
                "status_short": status.splitlines(),
            }
        except Exception:
            return {"commit": None, "dirty": None, "status_short": None}

    def _artifact_paths(self) -> Dict[str, object]:
        return {
            "logs_dir": str(self.log_dir),
            "visualizations_dir": str(self.vis_dir),
            "reports_dir": str(self.report_dir),
            "best_checkpoint": str(self.ckpt_dir / "best.pth"),
            "latest_checkpoint": str(self.ckpt_dir / "latest.pth"),
            "metrics_history": str(self.metrics_history_path),
            "resolved_config": str(self.resolved_config_path_json),
            "failure_report": str(self.report_dir / "failure.json"),
        }

    def _append_metrics_history(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        payload = {
            "epoch": epoch,
            "global_step": self.global_step,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "train": self._safe_jsonify(train_metrics),
            "val": self._safe_jsonify(val_metrics or {}),
            "best_epoch": self.best_epoch,
            "best_metric_name": self.best_metric_name,
            "best_selection_score": self.best_selection_score,
            "best_selection_value": self.best_selection_value,
        }
        with open(self.metrics_history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def _write_failure_report(self, exc: BaseException) -> None:
        failure = {
            "exp_name": getattr(self.cfg, "exp_name", self.output_dir.name),
            "status": "failed",
            "global_step": self.global_step,
            "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "artifacts": self._artifact_paths(),
        }
        self.failure_info = {
            "error_type": failure["error_type"],
            "error_message": failure["error_message"],
        }
        with open(self.report_dir / "failure.json", "w", encoding="utf-8") as f:
            json.dump(failure, f, indent=2)

    def _write_run_manifest(self) -> None:
        config_payload = self._safe_jsonify(self._config_to_dict())
        manifest = {
            "exp_name": getattr(self.cfg, "exp_name", self.output_dir.name),
            "output_dir": str(self.output_dir),
            "command": " ".join(sys.argv),
            "cwd": os.getcwd(),
            "pid": os.getpid(),
            "start_time": self.run_start_time,
            "hostname": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "config_path": self.resolved_config_path,
            "config": config_payload,
            "git": self._get_git_metadata(),
            "environment": {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "device": str(self.device),
            },
            "artifacts": self._artifact_paths(),
        }
        with open(self.report_dir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        with open(self.resolved_config_path_json, "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2)

    def _write_experiment_report(self, epoch: int, final: bool = False) -> None:
        duration_sec = max(0.0, time.time() - self.start_time_unix)
        report = {
            "exp_name": getattr(self.cfg, "exp_name", self.output_dir.name),
            "status": self.run_status,
            "epoch": epoch,
            "final": final,
            "global_step": self.global_step,
            "best_epoch": self.best_epoch,
            "best_metric_name": self.best_metric_name,
            "best_metric_mode": self.best_metric_mode,
            "best_selection_score": self.best_selection_score,
            "best_selection_value": self.best_selection_value,
            "last_train_metrics": self.last_train_metrics,
            "last_val_metrics": self.last_val_metrics,
            "best_checkpoint": str(self.ckpt_dir / "best.pth"),
            "latest_checkpoint": str(self.ckpt_dir / "latest.pth"),
            "artifacts": self._artifact_paths(),
            "start_time": self.run_start_time,
            "duration_sec": duration_sec,
            "failure": self.failure_info,
            "config_path": self.resolved_config_path,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.report_dir / "experiment_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    def _warmstart_refiner_state(
        self, refiner_state_dict: Dict[str, torch.Tensor]
    ) -> None:
        """Warmstart refiner weights when guide-channel count changes.

        V9->V10 style upgrades expand the first refiner conv from
        latent+RGB to latent+RGB+depth-guide channels. We preserve the learned
        V9 mapping for overlapping channels and zero-init only the newly added
        guide channels instead of restarting the full refiner.
        """
        if self.refiner is None:
            return

        current_state = self.refiner.state_dict()
        exact_loaded = 0
        partial_loaded = 0
        skipped: list[str] = []

        for key, source in refiner_state_dict.items():
            if key not in current_state:
                skipped.append(f"{key}:missing")
                continue

            target = current_state[key]
            if source.shape == target.shape:
                current_state[key] = source
                exact_loaded += 1
                continue

            if (
                key == "net.0.weight"
                and source.ndim == 4
                and target.ndim == 4
                and source.shape[0] == target.shape[0]
                and source.shape[2:] == target.shape[2:]
            ):
                copy_channels = min(source.shape[1], target.shape[1])
                patched = target.clone()
                patched.zero_()
                patched[:, :copy_channels] = source[:, :copy_channels]
                current_state[key] = patched
                partial_loaded += 1
                skipped.append(
                    f"{key}:partial {tuple(source.shape)} -> {tuple(target.shape)}"
                )
                continue

            skipped.append(f"{key}:{tuple(source.shape)} -> {tuple(target.shape)}")

        self.refiner.load_state_dict(current_state, strict=False)
        self._log(
            f"Warmstarted refiner with {exact_loaded} exact tensors and "
            f"{partial_loaded} partial tensor(s)"
        )
        if skipped:
            preview = ", ".join(skipped[:4])
            if len(skipped) > 4:
                preview += ", ..."
            self._log(f"Refiner warmstart skipped/mismatched: {preview}")

    def _warmstart_module_state(
        self,
        module: nn.Module,
        module_state_dict: Dict[str, torch.Tensor],
        module_name: str,
    ) -> None:
        """Warmstart only the exact-shape tensors for an upgraded module."""
        current_state = module.state_dict()
        exact_loaded = 0
        remapped_loaded = 0
        skipped: list[str] = []

        for key, source in module_state_dict.items():
            if key not in current_state:
                skipped.append(f"{key}:missing")
                continue

            target = current_state[key]
            if source.shape == target.shape:
                current_state[key] = source
                exact_loaded += 1
            else:
                skipped.append(f"{key}:{tuple(source.shape)} -> {tuple(target.shape)}")

        if module_name == "model" and getattr(self.cfg, "hybrid_decoupled_heads", False):
            old_fuse_prefix = "fusion_head.fuse."
            for suffix in ("0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias"):
                source_key = old_fuse_prefix + suffix
                if source_key not in module_state_dict:
                    continue
                source = module_state_dict[source_key]
                for branch in ("geometry_head", "semantic_head"):
                    target_key = f"fusion_head.{branch}.{suffix}"
                    target = current_state.get(target_key)
                    if target is not None and source.shape == target.shape:
                        current_state[target_key] = source
                        remapped_loaded += 1

            gate_stem = "fusion_head.gate.0"
            for suffix in ("weight", "bias"):
                source_key = f"{gate_stem}.{suffix}"
                source = module_state_dict.get(source_key)
                if source is None:
                    continue
                for branch in ("geometry_gate", "semantic_gate"):
                    target_key = f"fusion_head.{branch}.0.{suffix}"
                    target = current_state.get(target_key)
                    if target is not None and source.shape == target.shape:
                        current_state[target_key] = source
                        remapped_loaded += 1

            for suffix in ("weight", "bias"):
                source_key = f"fusion_head.gate.2.{suffix}"
                source = module_state_dict.get(source_key)
                if source is None:
                    continue
                if suffix == "weight" and source.ndim == 4:
                    source_reduced = source.mean(dim=0, keepdim=True)
                elif suffix == "bias" and source.ndim == 1:
                    source_reduced = source.mean(dim=0, keepdim=True)
                else:
                    source_reduced = source
                for branch in ("geometry_gate", "semantic_gate"):
                    target_key = f"fusion_head.{branch}.2.{suffix}"
                    target = current_state.get(target_key)
                    if target is not None and source_reduced.shape == target.shape:
                        current_state[target_key] = source_reduced
                        remapped_loaded += 1

            source_key = "fusion_head.fuse.0.weight"
            target_key = "fusion_head.fuse.0.weight"
            source = module_state_dict.get(source_key)
            target = current_state.get(target_key)
            if (
                source is not None
                and target is not None
                and source.ndim == 4
                and target.ndim == 4
                and source.shape[0] == target.shape[0]
                and source.shape[2:] == target.shape[2:]
                and target.shape[1] == source.shape[1] * 2
            ):
                patched = target.clone()
                patched.zero_()
                patched[:, :source.shape[1]] = source * 0.5
                patched[:, source.shape[1]: source.shape[1] * 2] = source * 0.5
                current_state[target_key] = patched
                remapped_loaded += 1

        module.load_state_dict(current_state, strict=False)
        self._log(
            f"Warmstarted {module_name} with {exact_loaded} exact tensors"
            + (f" and {remapped_loaded} remapped tensor(s)" if remapped_loaded else "")
        )
        if skipped:
            preview = ", ".join(skipped[:4])
            if len(skipped) > 4:
                preview += ", ..."
            self._log(f"{module_name} warmstart skipped/mismatched: {preview}")

    def load_checkpoint(self, path: str, resume: bool = True) -> None:
        checkpoint_map_location = self.device if resume else "cpu"
        ckpt = load_trusted_checkpoint(path, map_location=checkpoint_map_location)
        try:
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        except RuntimeError as e:
            self._log(
                f"Model state_dict size mismatch, attempting partial warmstart: {e}"
            )
            self._warmstart_module_state(
                self.model, ckpt["model_state_dict"], "model"
            )
        if "codec_state_dict" in ckpt:
            try:
                self.codec.load_state_dict(ckpt["codec_state_dict"], strict=False)
            except RuntimeError as e:
                self._log(
                    f"Codec state_dict size mismatch, attempting partial warmstart: {e}"
                )
                self._warmstart_module_state(
                    self.codec, ckpt["codec_state_dict"], "codec"
                )
        if "sharpener_state_dict" in ckpt:
            try:
                self.sharpener.load_state_dict(
                    ckpt["sharpener_state_dict"], strict=False
                )
            except RuntimeError as e:
                self._log(
                    f"Sharpener state_dict size mismatch, attempting partial warmstart: {e}"
                )
                self._warmstart_module_state(
                    self.sharpener, ckpt["sharpener_state_dict"], "sharpener"
                )
        if "refiner_state_dict" in ckpt and self.use_refiner and self.refiner is not None:
            try:
                self.refiner.load_state_dict(
                    ckpt["refiner_state_dict"], strict=False
                )
            except RuntimeError as e:
                self._log(
                    f"Refiner state_dict size mismatch, attempting partial warmstart: {e}"
                )
                self._warmstart_refiner_state(ckpt["refiner_state_dict"])
        if "depth_head_state_dict" in ckpt and self.depth_head is not None:
            try:
                self.depth_head.load_state_dict(
                    ckpt["depth_head_state_dict"], strict=False
                )
            except RuntimeError as e:
                self._log(
                    f"Depth head state_dict size mismatch, starting depth head from scratch: {e}"
                )
        if "seg_head_state_dict" in ckpt and self.seg_head is not None:
            try:
                self.seg_head.load_state_dict(
                    ckpt["seg_head_state_dict"], strict=False
                )
            except RuntimeError as e:
                self._log(
                    f"Seg head state_dict size mismatch, starting seg head from scratch: {e}"
                )
        if (
            "point_summary_adapter_state_dict" in ckpt
            and self.point_summary_adapter is not None
        ):
            try:
                self.point_summary_adapter.load_state_dict(
                    ckpt["point_summary_adapter_state_dict"], strict=False
                )
                self.point_summary_adapter_metadata = dict(
                    ckpt.get("point_summary_adapter_metadata") or {}
                )
                self.point_summary_adapter_epoch = ckpt.get(
                    "point_summary_adapter_epoch"
                )
                raw_metric = ckpt.get("point_summary_adapter_best_metric")
                self.point_summary_adapter_best_metric = (
                    float(raw_metric) if raw_metric is not None else None
                )
            except RuntimeError as e:
                self._log(
                    "Point summary adapter state_dict size mismatch, "
                    f"starting adapter from scratch: {e}"
                )
        if (
            "foundation_cache_projectors_state_dict" in ckpt
            and self.foundation_cache_projectors
        ):
            try:
                self.foundation_cache_projectors.load_state_dict(
                    ckpt["foundation_cache_projectors_state_dict"], strict=False
                )
            except RuntimeError as e:
                self._log(
                    "Foundation cache projector state_dict size mismatch, "
                    f"starting projector from scratch: {e}"
                )

        if resume:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except (ValueError, KeyError) as e:
                self._log(f"Optimizer state mismatch (new param groups?), "
                          f"re-initializing optimizer: {e}")
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except (ValueError, KeyError) as e:
                self._log(f"Scheduler state mismatch, re-initializing: {e}")
            if "scaler_state_dict" in ckpt:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.global_step = ckpt.get("global_step", 0)
            self.best_cosine = ckpt.get("best_cosine", -1.0)
            self.best_metric_name = ckpt.get("best_metric_name", self.best_metric_name)
            self.best_metric_mode = ckpt.get("best_metric_mode", self.best_metric_mode)
            if "best_selection_score" in ckpt:
                self.best_selection_score = ckpt["best_selection_score"]
            elif self.best_metric_name == "cosine":
                self.best_selection_score = self.best_cosine
            self.best_selection_value = ckpt.get("best_selection_value")
            self._log(f"Resumed from epoch {self.start_epoch - 1}")
        else:
            self._log(f"Warmstart: loaded model weights from {path}")
