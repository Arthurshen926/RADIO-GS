"""Fail-closed boundary for exact MoGe-3 inference.

The adapter never aliases another MoGe release to MoGe-3.  Callers must supply
a checkpoint, its digest, and a loader whose model receipt explicitly names the
requested family.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from radio_gs.v4.contracts.geometry_receipt import sha256_file


@dataclass(frozen=True)
class GeometryPrediction:
    point_map: torch.Tensor
    normals: torch.Tensor
    validity: torch.Tensor
    confidence: torch.Tensor
    intrinsics: torch.Tensor | None = None


class ExactMoge3Runner:
    FAMILY = "MoGe-3"

    def __init__(
        self,
        checkpoint: str | Path,
        loader: Callable[[Path], Any],
        *,
        declared_family: str,
    ) -> None:
        if declared_family != self.FAMILY:
            raise ValueError("exact MoGe-3 runner refuses checkpoints from another model family")
        self.checkpoint = Path(checkpoint).resolve(strict=True)
        self.checkpoint_sha256 = sha256_file(self.checkpoint)
        self.model = loader(self.checkpoint)

    @torch.no_grad()
    def predict(self, rgb: torch.Tensor, *, fov_x: float | None = None) -> GeometryPrediction:
        output = self.model(torch.as_tensor(rgb))
        required = {"point_map", "normals", "validity", "confidence"}
        if not isinstance(output, dict) or not required.issubset(output):
            raise ValueError("MoGe-3 adapter output lacks required geometry fields")
        prediction = GeometryPrediction(**{name: torch.as_tensor(output[name]).cpu() for name in required})
        height, width = prediction.validity.shape
        if prediction.point_map.shape != (height, width, 3) or prediction.normals.shape != (height, width, 3):
            raise ValueError("MoGe-3 point map and normals must have shape [H, W, 3]")
        if prediction.confidence.shape != (height, width):
            raise ValueError("MoGe-3 confidence must match validity")
        return prediction


class OfficialMoge3Runner:
    """Revision-pinned adapter for the released Microsoft/Ruicheng MoGe-3.

    MoGe-3 exposes a binary geometry-validity mask rather than a separate
    calibrated confidence map.  ``confidence`` therefore records that mask as
    an observation weight; it must not be described as learned uncertainty.
    """

    FAMILY = "MoGe-3"
    RELEASES = {
        "Ruicheng/moge-3-vitl": "a96e58bad16a94c9a3c193a5d4cd75b4b6906c94",
        "Ruicheng/moge-3-vitg": "9d2e5caca41d40e86843c18802f861c8f2091ddb",
    }

    def __init__(
        self,
        model: Any,
        checkpoint: str | Path,
        *,
        model_id: str,
        revision: str,
        device: str | torch.device,
        resolution_level: int = 9,
        refine_steps: int = 3,
        use_fp16: bool = True,
    ) -> None:
        expected_revision = self.RELEASES.get(model_id)
        if expected_revision is None or revision != expected_revision:
            raise ValueError("MoGe-3 model id and revision must match the audited release allowlist")
        self.model_id = model_id
        self.revision = revision
        self.device = torch.device(device)
        self.checkpoint = Path(checkpoint).resolve(strict=True)
        self.checkpoint_sha256 = sha256_file(self.checkpoint)
        self.resolution_level = int(resolution_level)
        self.refine_steps = int(refine_steps)
        self.use_fp16 = bool(use_fp16)
        self.model = model.to(self.device).eval()

    @classmethod
    def from_huggingface(
        cls,
        model_id: str,
        *,
        revision: str,
        device: str | torch.device,
        resolution_level: int = 9,
        refine_steps: int = 3,
        use_fp16: bool = True,
    ) -> "OfficialMoge3Runner":
        expected_revision = cls.RELEASES.get(model_id)
        if expected_revision is None or revision != expected_revision:
            raise ValueError("MoGe-3 download must use an audited model id and exact revision")
        try:
            from huggingface_hub import hf_hub_download
            from moge.model.v3 import MoGeModel
        except ImportError as exc:
            raise RuntimeError(
                "official MoGe-3 dependencies are missing; install the pinned Microsoft MoGe repository"
            ) from exc
        checkpoint = hf_hub_download(repo_id=model_id, filename="model.pt", revision=revision)
        model = MoGeModel.from_pretrained(checkpoint)
        return cls(
            model,
            checkpoint,
            model_id=model_id,
            revision=revision,
            device=device,
            resolution_level=resolution_level,
            refine_steps=refine_steps,
            use_fp16=use_fp16,
        )

    @torch.inference_mode()
    def predict(self, rgb: torch.Tensor, *, fov_x: float | None = None) -> GeometryPrediction:
        image = torch.as_tensor(rgb, dtype=torch.float32, device=self.device)
        if image.ndim != 3:
            raise ValueError("MoGe-3 RGB input must have shape [3, H, W] or [H, W, 3]")
        if image.shape[0] != 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
        if image.shape[0] != 3:
            raise ValueError("MoGe-3 RGB input must have exactly three channels")
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("MoGe-3 RGB input must be finite and normalized to [0, 1]")

        output = self.model.infer(
            image,
            resolution_level=self.resolution_level,
            refine_steps=self.refine_steps,
            use_fp16=self.use_fp16,
            force_projection=True,
            apply_mask=False,
            fov_x=fov_x,
        )
        required = {"points", "normal", "mask"}
        if not isinstance(output, dict) or not required.issubset(output):
            raise ValueError("official MoGe-3 output lacks points, normal, or mask")
        validity = torch.as_tensor(output["mask"], dtype=torch.bool)
        point_map = torch.as_tensor(output["points"])
        normals = torch.as_tensor(output["normal"])
        height, width = validity.shape
        if point_map.shape != (height, width, 3) or normals.shape != (height, width, 3):
            raise ValueError("official MoGe-3 geometry maps must have shape [H, W, 3]")
        validity = validity & torch.isfinite(point_map).all(dim=-1) & (point_map[..., 2] > 0)
        return GeometryPrediction(
            point_map=point_map.float().cpu(),
            normals=normals.float().cpu(),
            validity=validity.cpu(),
            confidence=validity.float().cpu(),
            intrinsics=(
                torch.as_tensor(output["intrinsics"]).float().cpu()
                if "intrinsics" in output
                else None
            ),
        )
