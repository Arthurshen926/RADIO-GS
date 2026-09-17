"""Read sealed calibrated source depth for the Gaussian semantic map builder."""
from __future__ import annotations

import json
import math
from pathlib import Path
import torch

from radio_gs.field.geometry_prior import CONTRACT, depth_conditioned_weights
from radio_gs.utils.immutable_artifacts import sha256_file


class MogeGeometryPrior:
    def __init__(self, report_path: str, expected_sha256: str, *, strength: float = 0.5):
        if not math.isfinite(strength) or not 0 <= strength < 1:
            raise ValueError("geometry strength must be in [0,1)")
        self.path = Path(report_path).resolve(strict=True)
        if not expected_sha256 or sha256_file(self.path) != expected_sha256:
            raise ValueError("MoGe calibration report digest differs")
        self.report = json.loads(self.path.read_text())
        receipt = self.report["geometry_receipt"]
        if any(receipt.get(k, False) for k in (
            "target_rgb_opened", "benchmark_images_opened", "benchmark_masks_opened", "benchmark_labels_opened"
        )):
            raise ValueError("geometry prior must be source-only")
        # Verify the upstream source/model/calibration inputs, not only report bytes.
        inputs = receipt["inputs"]
        by_role = {row["role"]: row for row in inputs}
        for row in inputs:
            if row["role"] == "fused_surface":
                continue  # The semantic map consumes depth, not fused voxels.
            if sha256_file(row["path"]) != row["sha256"]:
                raise ValueError("MoGe calibration input digest differs")
        manifest_path = Path(by_role["moge3_scene_manifest"]["path"])
        self.manifest = json.loads(manifest_path.read_text())
        self.records = {int(r["frame_index"]): r for r in self.manifest["records"]}
        self.calibration = {int(r["frame_index"]): r for r in self.report["calibration"]}
        authority = json.loads(Path(by_role["source_authority"]["path"]).read_text())
        if set(self.records) != set(map(int, authority["frame_indices"])):
            raise ValueError("MoGe source frame authority differs")
        from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_images, _read_cameras_binary
        self.cameras = _read_images(Path(by_role["colmap_images"]["path"]),
                                    _read_cameras_binary(Path(by_role["colmap_cameras"]["path"])),
                                    authority.get("camera_name_to_frame_index"))
        self.scale = float(self.report["global_colmap_units_per_metre"])
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("invalid MoGe world scale")
        self.source_authority_sha256 = by_role["source_authority"]["sha256"]
        self.strength = strength
        self.statistics = []

    def apply(self, *, frame_index, xyz, world_to_camera, intrinsic, gaussian_ids, pixel_ids, weights, height, width):
        if frame_index not in self.records:
            raise ValueError(f"Gaussian source frame has no MoGe authority: {frame_index}")
        camera = self.cameras[frame_index]
        if not torch.allclose(world_to_camera.float().cpu(), camera.world_to_camera, atol=1e-4, rtol=1e-5):
            raise ValueError("Gaussian and MoGe camera coordinate frames differ")
        k = intrinsic.float().cpu()
        if k.shape != (3, 3) or not bool(torch.isfinite(k).all()) or k[0, 0] <= 0 or k[1, 1] <= 0:
            raise ValueError("invalid Gaussian intrinsics")
        row = self.records[frame_index]
        prediction = Path(row["prediction"])
        if sha256_file(prediction) != row["prediction_sha256"]:
            raise ValueError("MoGe prediction digest differs")
        calibration = self.calibration[frame_index]
        accepted = bool(calibration["accepted"])
        if calibration.get('source_geometry_override') is not None:
            from radio_gs.geometry.source_geometry_override import load_source_geometry_override
            materialized = load_source_geometry_override(calibration['source_geometry_override'], frame=frame_index,
                prediction_sha256=row['prediction_sha256'], camera=camera)
            depth, valid = materialized['depth'], materialized['valid']
        else:
            payload = torch.load(prediction, map_location="cpu", weights_only=False)
            if payload["model_checkpoint_sha256"] != self.manifest["model_checkpoint_sha256"]:
                raise ValueError("MoGe model identity differs")
            scale = float(calibration["scale"]) if accepted else 1.0
            offset = float(calibration["offset"]) if accepted else 0.0
            depth = torch.as_tensor(payload["point_map"])[..., 2].float() * self.scale * scale + offset
            valid = torch.as_tensor(payload["validity"]).bool()
        # Sample the same image-edge pixel ray at the teacher grid. No bilinear
        # mixing across a foreground/background discontinuity.
        pids = pixel_ids.long().cpu()
        pixels = torch.stack([(pids % width).float() + 0.5,
                              torch.div(pids, width, rounding_mode="floor").float() + 0.5,
                              torch.ones(pids.numel())], dim=1)
        source_k = camera.intrinsic.clone()
        source_k[0] *= depth.shape[1] / camera.width
        source_k[1] *= depth.shape[0] / camera.height
        projected = pixels @ torch.linalg.inv(k).T @ source_k.T
        x = torch.floor(projected[:, 0] / projected[:, 2]).long()
        y = torch.floor(projected[:, 1] / projected[:, 2]).long()
        inside = (x >= 0) & (x < depth.shape[1]) & (y >= 0) & (y < depth.shape[0])
        x, y = x.clamp(0, depth.shape[1]-1), y.clamp(0, depth.shape[0]-1)
        d = depth[y, x].to(weights.device)
        c = (valid[y, x] & inside).to(weights).float() * float(accepted)
        pose = world_to_camera.to(device=weights.device, dtype=torch.float32)
        points = xyz[gaussian_ids.long().cpu()].to(weights.device)
        z = points @ pose[2, :3] + pose[2, 3]
        revised, known = depth_conditioned_weights(
            weights, z, d, c, absolute_tolerance=0.08 * self.scale,
            relative_tolerance=0.02, strength=self.strength,
        )
        self.statistics.append({
            "frame_index": frame_index, "hits": weights.numel(),
            "known_depth_hits": int(known.sum()), "calibration_accepted": accepted,
            "original_mass": float(weights.sum()), "conditioned_mass": float(revised.sum()),
            "unique_gaussians": int(gaussian_ids.unique().numel()),
            "median_gaussian_depth_metres": float(z.median() / self.scale) if z.numel() else None,
            "median_source_depth_metres": float(d[known].median() / self.scale) if bool(known.any()) else None,
            "saturated_residual_fraction": float((revised <= weights * (1 - self.strength + 1e-6)).float().mean()) if weights.numel() and self.strength > 0 else 0.0,
        })
        return revised

    def metadata(self):
        return {"contract": CONTRACT, "report": str(self.path), "report_sha256": sha256_file(self.path),
                "strength": self.strength, "absolute_tolerance_metres": 0.08,
                "relative_tolerance": 0.02, "query_independent": True,
                "compositor_authority_modified": False, "centre_depth_approximation": True,
                "unknown_policy": "retain_original_write", "views": self.statistics}
