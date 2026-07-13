"""Official, frozen capability views derived from canonical RADIO features."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from radio_gs.models.radio_adaptors import load_radio_adaptor_from_checkpoint
from radio_gs.models.siglip_projection import (
    SigLIP2FeatureProjection,
    SigLIP2SummaryHead,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


class FrozenRadioViews(nn.Module):
    """Frozen C-RADIOv4 adaptor modules with explicit validity boundaries.

    DINOv3 and SAM3 feature projections are pointwise MLPs and may be applied
    to canonical primitive RADIO rows.  ``siglip2-g`` spatial projection has
    attention over a declared 2-D token set; it is therefore exposed only as a
    contextual-token operation.  It must not be applied to arbitrary primitive
    chunks and called a canonical descriptor.
    """

    def __init__(
        self,
        *,
        siglip_spatial: nn.Module,
        siglip_summary: nn.Module,
        dino: nn.Module,
        sam3: nn.Module,
        radio_checkpoint_sha256: str,
    ) -> None:
        super().__init__()
        self.siglip_spatial = _freeze(siglip_spatial)
        self.siglip_summary = _freeze(siglip_summary)
        self.dino = _freeze(dino)
        self.sam3 = _freeze(sam3)
        self.radio_checkpoint_sha256 = str(radio_checkpoint_sha256)

    @classmethod
    def from_radio_checkpoint(cls, checkpoint_path: str | Path) -> "FrozenRadioViews":
        path = Path(checkpoint_path)
        return cls(
            siglip_spatial=SigLIP2FeatureProjection.from_radio_checkpoint(str(path)),
            siglip_summary=SigLIP2SummaryHead.from_radio_checkpoint(str(path)),
            dino=load_radio_adaptor_from_checkpoint(path, "dino_v3", kind="feature_projection"),
            sam3=load_radio_adaptor_from_checkpoint(path, "sam3", kind="feature_projection"),
            radio_checkpoint_sha256=sha256_file(path),
        )

    def project_siglip_spatial_tokens(
        self,
        tokens: torch.Tensor,
        *,
        token_layout: tuple[int, int],
    ) -> torch.Tensor:
        """Official level-1 spatial oracle on one complete 2-D token grid."""

        if tokens.ndim != 3 or tokens.shape[-1] != 1280:
            raise ValueError("tokens must be [B,N,1280]")
        height, width = (int(v) for v in token_layout)
        if height <= 0 or width <= 0 or height * width != tokens.shape[1]:
            raise ValueError("token_layout must exactly match the complete token set")
        return F.normalize(self.siglip_spatial(tokens).float(), dim=-1, eps=1e-8)

    def project_dino_primitives(self, radio_features: torch.Tensor) -> torch.Tensor:
        if radio_features.ndim != 2 or radio_features.shape[1] != 1280:
            raise ValueError("radio_features must be [N,1280]")
        return F.normalize(self.dino(radio_features[None])[0].float(), dim=-1, eps=1e-8)

    def project_sam3_primitives(self, radio_features: torch.Tensor) -> torch.Tensor:
        if radio_features.ndim != 2 or radio_features.shape[1] != 1280:
            raise ValueError("radio_features must be [N,1280]")
        return F.normalize(self.sam3(radio_features[None])[0].float(), dim=-1, eps=1e-8)

    def project_official_summary_token(self, summary_token: torch.Tensor) -> torch.Tensor:
        """Project genuine RADIO summary tokens, never pooled spatial tokens."""

        values = summary_token
        if values.ndim == 2:
            values = values[:, None, :]
        if values.ndim != 3 or values.shape[-1] != 1280:
            raise ValueError("summary_token must be [B,1280] or [B,S,1280]")
        return F.normalize(self.siglip_summary(values).float(), dim=-1, eps=1e-8)


@dataclass
class OfficialRadioRuntime:
    """Thin wrapper around the official TorchHub C-RADIO runtime."""

    model: nn.Module
    version: str
    adaptor_names: tuple[str, ...]

    @classmethod
    def load(
        cls,
        *,
        radio_repo: str = "/root/RADIO",
        version: str = "c-radio_v4-h",
        adaptor_names: Iterable[str] = ("siglip2-g", "dino_v3", "sam3"),
        device: str | torch.device = "cuda",
    ) -> "OfficialRadioRuntime":
        names = tuple(dict.fromkeys(str(name) for name in adaptor_names))
        model = torch.hub.load(
            radio_repo,
            "radio_model",
            source="local",
            version=version,
            progress=True,
            skip_validation=True,
            adaptor_names=list(names),
        )
        model = _freeze(model).to(device)
        return cls(model=model, version=version, adaptor_names=names)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor, *, feature_fmt: str = "NCHW"):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must be [B,3,H,W] in [0,1]")
        nearest = self.model.get_nearest_supported_resolution(*images.shape[-2:])
        target_size = (int(nearest.height), int(nearest.width))
        if tuple(images.shape[-2:]) != target_size:
            images = F.interpolate(
                images, target_size, mode="bilinear", align_corners=False
            )
        return self.model(images, feature_fmt=feature_fmt)

    @torch.no_grad()
    def encode_adaptor_images(
        self,
        images: torch.Tensor,
        adaptor_name: str,
        *,
        feature_fmt: str = "NCHW",
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Return one official adaptor's summary and spatial output.

        C-RADIO releases have used tuples, named tuples, and dictionaries for
        adaptor outputs.  Keeping that compatibility here prevents query
        front-ends from reimplementing (or accidentally replacing) an official
        adaptor head.
        """

        output = self.encode_images(images, feature_fmt=feature_fmt)
        if not isinstance(output, Mapping) or adaptor_name not in output:
            raise RuntimeError(
                f"official runtime did not return {adaptor_name!r} adaptor output"
            )
        value = output[adaptor_name]
        if isinstance(value, Mapping):
            summary = value.get("summary")
            spatial = value.get("features")
            if spatial is None:
                spatial = value.get("spatial")
        elif hasattr(value, "features"):
            summary = getattr(value, "summary", None)
            spatial = value.features
        elif isinstance(value, (tuple, list)) and len(value) >= 2:
            summary, spatial = value[0], value[1]
        else:
            raise TypeError(
                f"unsupported official adaptor output type: {type(value)!r}"
            )
        if spatial is None or not torch.is_tensor(spatial):
            raise RuntimeError(f"{adaptor_name!r} adaptor has no spatial output")
        return summary, spatial

    @torch.no_grad()
    def encode_text(self, texts: Iterable[str]) -> torch.Tensor:
        if "siglip2-g" not in self.model.adaptors:
            raise RuntimeError("official runtime was not loaded with siglip2-g")
        adaptor = self.model.adaptors["siglip2-g"]
        text_list = [str(text) for text in texts]
        if not text_list:
            raise ValueError("texts cannot be empty")
        tokens = adaptor.tokenizer(text_list)
        first = next(adaptor.parameters())
        tokens = tokens.to(first.device)
        return adaptor.encode_text(tokens, normalize=True).float()

    @torch.no_grad()
    def encode_official_crop_summaries(self, crops: torch.Tensor) -> torch.Tensor:
        """Level-2 target: official visual summary from re-encoded crops."""

        output = self.encode_images(crops, feature_fmt="NCHW")
        if not isinstance(output, Mapping) or "siglip2-g" not in output:
            raise RuntimeError("official runtime did not return siglip2-g adaptor output")
        summary = output["siglip2-g"].summary
        return F.normalize(summary.float(), dim=-1, eps=1e-8)


@dataclass
class OfficialCropSummaryRuntime:
    """Official C-RADIO backbone plus its frozen siglip2-g visual summary head.

    This avoids loading a second HF vision tower: crops are encoded by
    C-RADIOv4, the genuine SigLIP teacher summary slot is selected, and the
    official ``_heads.siglip2-g`` checkpoint module maps it to text space.
    """

    backbone: nn.Module
    summary_head: nn.Module
    version: str
    radio_checkpoint_sha256: str

    @classmethod
    def load(
        cls,
        *,
        checkpoint_path: str | Path,
        radio_repo: str = "/root/RADIO",
        version: str = "c-radio_v4-h",
        device: str | torch.device = "cuda",
    ) -> "OfficialCropSummaryRuntime":
        backbone = torch.hub.load(
            radio_repo,
            "radio_model",
            source="local",
            version=version,
            progress=True,
            skip_validation=True,
            adaptor_names=[],
        )
        backbone = _freeze(backbone).to(device)
        summary_head = _freeze(
            SigLIP2SummaryHead.from_radio_checkpoint(checkpoint_path)
        ).to(device)
        return cls(
            backbone=backbone,
            summary_head=summary_head,
            version=version,
            radio_checkpoint_sha256=sha256_file(checkpoint_path),
        )

    @torch.no_grad()
    def encode(self, crops: torch.Tensor) -> torch.Tensor:
        if crops.ndim != 4 or crops.shape[1] != 3:
            raise ValueError("crops must be [B,3,H,W] in [0,1]")
        nearest = self.backbone.get_nearest_supported_resolution(*crops.shape[-2:])
        target_size = (int(nearest.height), int(nearest.width))
        if tuple(crops.shape[-2:]) != target_size:
            crops = F.interpolate(crops, target_size, mode="bilinear", align_corners=False)
        output = self.backbone(crops)
        summary = output.summary if hasattr(output, "summary") else output[0]
        if summary.ndim != 2 or summary.shape[1] % 1280 != 0:
            raise RuntimeError("unexpected C-RADIO summary layout")
        teacher_slots = summary.reshape(summary.shape[0], -1, 1280)
        siglip_summary_token = teacher_slots[:, 0]
        descriptor = self.summary_head(siglip_summary_token[:, None])[:, 0]
        return F.normalize(descriptor.float(), dim=-1, eps=1e-8)

    @torch.no_grad()
    def encode_training_pair(
        self, crops: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return raw spatial tokens, official summary token, and descriptor."""

        if crops.ndim != 4 or crops.shape[1] != 3:
            raise ValueError("crops must be [B,3,H,W] in [0,1]")
        nearest = self.backbone.get_nearest_supported_resolution(*crops.shape[-2:])
        target_size = (int(nearest.height), int(nearest.width))
        if tuple(crops.shape[-2:]) != target_size:
            crops = F.interpolate(crops, target_size, mode="bilinear", align_corners=False)
        output = self.backbone(crops, feature_fmt="NCHW")
        summary = output.summary if hasattr(output, "summary") else output[0]
        spatial = output.features if hasattr(output, "features") else output[1]
        if spatial.ndim != 4 or spatial.shape[1] != 1280:
            raise RuntimeError("unexpected C-RADIO spatial layout")
        if summary.ndim != 2 or summary.shape[1] % 1280 != 0:
            raise RuntimeError("unexpected C-RADIO summary layout")
        summary_token = summary.reshape(summary.shape[0], -1, 1280)[:, 0]
        descriptor = self.summary_head(summary_token[:, None])[:, 0]
        return (
            spatial.float(),
            summary_token.float(),
            F.normalize(descriptor.float(), dim=-1, eps=1e-8),
        )
