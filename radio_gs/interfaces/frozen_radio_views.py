"""Official, frozen capability views derived from canonical RADIO features."""

from __future__ import annotations

from dataclasses import dataclass
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
from radio_gs.utils.immutable_artifacts import sha256_file


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
    def from_radio_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> "FrozenRadioViews":
        path = Path(checkpoint_path)
        checkpoint_sha256 = str(expected_sha256 or sha256_file(path))
        return cls(
            siglip_spatial=SigLIP2FeatureProjection.from_radio_checkpoint(
                str(path), expected_sha256=checkpoint_sha256
            ),
            siglip_summary=SigLIP2SummaryHead.from_radio_checkpoint(
                str(path), expected_sha256=checkpoint_sha256
            ),
            dino=load_radio_adaptor_from_checkpoint(
                path,
                "dino_v3",
                kind="feature_projection",
                expected_sha256=checkpoint_sha256,
            ),
            sam3=load_radio_adaptor_from_checkpoint(
                path,
                "sam3",
                kind="feature_projection",
                expected_sha256=checkpoint_sha256,
            ),
            radio_checkpoint_sha256=checkpoint_sha256,
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

        summary, _ = self.encode_adaptor_images(crops, "siglip2-g")
        if (
            not torch.is_tensor(summary)
            or summary.shape != (crops.shape[0], 1536)
            or not bool(torch.isfinite(summary).all())
            or bool((summary.float().norm(dim=-1) <= 1e-8).any())
        ):
            raise RuntimeError("official SigLIP2 summary must be finite nonzero [B,1536]")
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
    summary_slot_index: int = 0

    @classmethod
    def load(
        cls,
        *,
        checkpoint_path: str | Path,
        radio_repo: str = "/root/RADIO",
        version: str = "c-radio_v4-h",
        device: str | torch.device = "cuda",
        parameter_dtype: torch.dtype | None = None,
    ) -> "OfficialCropSummaryRuntime":
        checkpoint_path = Path(checkpoint_path).resolve(strict=True)
        backbone, checkpoint = torch.hub.load(
            radio_repo,
            "radio_model",
            source="local",
            version=str(checkpoint_path),
            progress=True,
            skip_validation=True,
            adaptor_names=[],
            return_checkpoint=True,
        )
        teacher_args = checkpoint["args"]
        teachers = teacher_args.teachers
        siglip = [(index, teacher) for index, teacher in enumerate(teachers)
                  if teacher["name"] == "siglip2-g"]
        if len(siglip) != 1:
            raise ValueError("checkpoint must define exactly one SigLIP2 teacher")
        teacher_index, teacher = siglip[0]
        if getattr(teacher_args, "cls_token_per_teacher", True):
            name_to_index = {}
            for index, item in enumerate(teachers):
                if item.get("use_summary", True):
                    name_to_index.setdefault(item["name"], index)
            slots = sorted(name_to_index.values())
            teacher_slot = teacher.get("token_slot", teacher_index)
            if teacher_slot not in slots:
                raise ValueError("SigLIP2 summary token is absent from backbone output")
            summary_slot_index = slots.index(teacher_slot)
        else:
            summary_slot_index = 0
        del checkpoint
        backbone = _freeze(backbone)
        summary_head = _freeze(SigLIP2SummaryHead.from_radio_checkpoint(checkpoint_path))
        if parameter_dtype is not None:
            if parameter_dtype not in (torch.float16, torch.float32, torch.bfloat16):
                raise ValueError("crop-summary runtime requires a floating parameter dtype")
            backbone = backbone.to(dtype=parameter_dtype)
            summary_head = summary_head.to(dtype=parameter_dtype)
            # RADIO's input conditioner keeps its output dtype as a Python
            # attribute, so ``Module.to(dtype=...)`` cannot update it.  Without
            # this explicit synchronization it upcasts half inputs back to
            # float32 before the half-precision patch embedder.
            conditioner = getattr(backbone, "input_conditioner", None)
            if conditioner is not None and hasattr(conditioner, "dtype"):
                conditioner.dtype = parameter_dtype
        backbone = backbone.to(device)
        summary_head = summary_head.to(device)
        return cls(
            backbone=backbone,
            summary_head=summary_head,
            version=version,
            radio_checkpoint_sha256=sha256_file(checkpoint_path),
            summary_slot_index=summary_slot_index,
        )

    @torch.no_grad()
    def encode(self, crops: torch.Tensor) -> torch.Tensor:
        if crops.ndim != 4 or crops.shape[1] != 3:
            raise ValueError("crops must be [B,3,H,W] in [0,1]")
        nearest = self.backbone.get_nearest_supported_resolution(*crops.shape[-2:])
        target_size = (int(nearest.height), int(nearest.width))
        if tuple(crops.shape[-2:]) != target_size:
            crops = F.interpolate(crops, target_size, mode="bilinear", align_corners=False)
        parameter = next(self.backbone.parameters())
        crops = crops.to(device=parameter.device, dtype=parameter.dtype)
        output = self.backbone(crops)
        summary = output.summary if hasattr(output, "summary") else output[0]
        if summary.ndim != 2 or summary.shape[1] % 1280 != 0:
            raise RuntimeError("unexpected C-RADIO summary layout")
        teacher_slots = summary.reshape(summary.shape[0], -1, 1280)
        siglip_summary_token = teacher_slots[:, self.summary_slot_index]
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
        parameter = next(self.backbone.parameters())
        crops = crops.to(device=parameter.device, dtype=parameter.dtype)
        output = self.backbone(crops, feature_fmt="NCHW")
        summary = output.summary if hasattr(output, "summary") else output[0]
        spatial = output.features if hasattr(output, "features") else output[1]
        if spatial.ndim != 4 or spatial.shape[1] != 1280:
            raise RuntimeError("unexpected C-RADIO spatial layout")
        if summary.ndim != 2 or summary.shape[1] % 1280 != 0:
            raise RuntimeError("unexpected C-RADIO summary layout")
        summary_token = summary.reshape(summary.shape[0], -1, 1280)[:, self.summary_slot_index]
        descriptor = self.summary_head(summary_token[:, None])[:, 0]
        return (
            spatial.float(),
            summary_token.float(),
            F.normalize(descriptor.float(), dim=-1, eps=1e-8),
        )
