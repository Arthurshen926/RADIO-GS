"""Canonical query interface for the sealed SUGM-v3 D512+R5 scene state."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from radio_gs.v3.contracts.method import validate_scene_state
from radio_gs.v3.memory.structured_memory import LowRankPrivateBranchMemory, SharedPrivateLayout
from radio_gs.v3.query.membership import membership_from_prototype, pool_prototype
from radio_gs.v3.query.calibrated_posterior import NullCalibratedPosterior
from radio_gs.v3.query.identity_adapter import (
    AffineTextAlignment,
    DirectTextProjection,
    LowRankIdentityAdapter,
    OrthogonalTextAlignment,
)
from radio_gs.v3.query.packet import QueryPacket
from radio_gs.v3.training.instance_upper_bound import sha256_file
from radio_gs.v3.training.rendered_mask import render_membership


class StructuredGaussianQueryInterface(nn.Module):
    """Produce one Gaussian posterior and render it unchanged in every view."""

    def __init__(
        self,
        model: LowRankPrivateBranchMemory,
        reliability: torch.Tensor,
        boundary_head: nn.Linear,
        siglip_mean: torch.Tensor | None = None,
        siglip_basis: torch.Tensor | None = None,
        identity_adapter: LowRankIdentityAdapter | None = None,
        text_alignment: nn.Module | None = None,
        text_negative_tokens: torch.Tensor | None = None,
        text_logit_scale: float = 10.0,
        center_semantic_identity: bool = False,
    ) -> None:
        super().__init__()
        value = torch.as_tensor(reliability).float()
        if value.shape != (model.memory.shape[0], 5):
            raise ValueError("query reliability axes differ from the D512 field")
        self.model = model
        self.boundary_head = boundary_head
        self.identity_adapter = identity_adapter
        self.text_alignment = text_alignment
        if text_logit_scale <= 0:
            raise ValueError("text logit scale differs")
        self.text_logit_scale = float(text_logit_scale)
        self.center_semantic_identity = bool(center_semantic_identity)
        self.register_buffer("reliability", value, persistent=False)
        if (siglip_mean is None) != (siglip_basis is None):
            raise ValueError("semantic identity codec is incomplete")
        if siglip_mean is not None:
            mean = torch.as_tensor(siglip_mean).float()
            basis = torch.as_tensor(siglip_basis).float()
            if mean.shape != (1536,) or basis.shape != (1536, 128):
                raise ValueError("semantic identity codec axes differ")
            self.register_buffer("siglip_mean", mean, persistent=False)
            self.register_buffer("siglip_basis", basis, persistent=False)
        else:
            self.siglip_mean = None
            self.siglip_basis = None
        if text_negative_tokens is not None:
            negatives = torch.as_tensor(text_negative_tokens).float()
            if negatives.ndim != 2 or negatives.shape[1] != 1536 or not negatives.shape[0]:
                raise ValueError("canonical text negative axes differ")
            self.register_buffer("text_negative_tokens", negatives, persistent=False)
        else:
            self.text_negative_tokens = None

    def _project_semantic_tokens(self, tokens: torch.Tensor, *, text: bool) -> torch.Tensor:
        raw = torch.as_tensor(tokens, device=self.model.memory.device).float().reshape(-1, 1536)
        if text and isinstance(self.text_alignment, DirectTextProjection):
            value = self.text_alignment.project_raw(raw, self.siglip_mean)
        else:
            value = torch.nn.functional.normalize(
                (raw - self.siglip_mean) @ self.siglip_basis, dim=-1, eps=1e-8
            )
        if text and self.text_alignment is not None and not isinstance(
            self.text_alignment, DirectTextProjection
        ):
            value = self.text_alignment(value)
        if self.identity_adapter is not None:
            value = self.identity_adapter(value)
        return value

    def _center_identity(
        self, semantic: torch.Tensor, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.center_semantic_identity:
            return semantic, query
        known = semantic.norm(dim=1) > 1e-6
        if not bool(known.any()):
            raise ValueError("semantic centering has no known source-written rows")
        centroid = semantic[known].mean(dim=0)
        return (
            torch.nn.functional.normalize(semantic - centroid, dim=-1, eps=1e-8),
            torch.nn.functional.normalize(query - centroid, dim=-1, eps=1e-8),
        )

    @torch.no_grad()
    def semantic_text_evidence(
        self, packet: QueryPacket
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep positive, canonical-null, and unknown evidence disentangled.

        Canonical negatives are useful final-posterior evidence, but they must
        not silently replace the positive text score used to choose identity
        anchors.  Returning the three axes separately makes that distinction
        explicit and lets a source-trained calibrator decide how much null
        evidence should count.
        """

        if packet.modality not in ("text", "image"):
            raise ValueError("semantic evidence requires a text or image packet")
        if self.siglip_mean is None or self.siglip_basis is None:
            raise ValueError("scene state has no sealed semantic identity codec")
        query = self._project_semantic_tokens(
            packet.token, text=packet.modality == "text"
        )[0]
        original_semantic = self.model.semantic_view()
        unknown = original_semantic.norm(dim=1) <= 1e-6
        semantic, query = self._center_identity(original_semantic, query)
        positive = semantic @ query
        null = torch.zeros_like(positive)
        if packet.modality == "text" and self.text_negative_tokens is not None:
            negatives = self._project_semantic_tokens(
                self.text_negative_tokens, text=True
            )
            if self.center_semantic_identity:
                known = ~unknown
                centroid = original_semantic[known].mean(dim=0)
                negatives = torch.nn.functional.normalize(
                    negatives - centroid, dim=-1, eps=1e-8
                )
            null = (semantic @ negatives.T).max(dim=1).values
        return positive, null, unknown.float()

    @torch.no_grad()
    def compile_identity_anchors(
        self,
        packet: QueryPacket,
        *,
        topk: int = 8,
        text_anchor_policy: str = "null_adjusted",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if topk <= 0 or topk > self.model.memory.shape[0]:
            raise ValueError("identity anchor budget differs")
        if packet.modality in ("text", "image"):
            if text_anchor_policy not in ("positive", "null_adjusted"):
                raise ValueError("text anchor policy differs")
            positive, null, _unknown = self.semantic_text_evidence(packet)
            score = positive
            if (
                text_anchor_policy == "null_adjusted"
                and packet.modality == "text"
                and self.text_negative_tokens is not None
            ):
                score = torch.sigmoid((positive - null) * self.text_logit_scale)
        else:
            score = torch.as_tensor(
                packet.seed_probability, device=self.model.memory.device
            ).float().reshape(-1)
            if score.shape != (self.model.memory.shape[0],):
                raise ValueError("prompt seed row domain differs from the scene state")
            score = score.nan_to_num(nan=-torch.inf)
        values, rows = score.topk(topk)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("query has fewer finite anchors than requested")
        weights = torch.softmax(values - values.max(), dim=0)
        return rows, weights, score

    @torch.no_grad()
    def posterior_from_packet(
        self,
        packet: QueryPacket,
        *,
        scale: float,
        topk: int = 8,
        temperature: float = 0.15,
        membership_margin: float = 0.0,
        identity_extent_weight: float = 0.0,
        posterior_chunk_size: int = 0,
        text_anchor_policy: str = "null_adjusted",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows, weights, identity = self.compile_identity_anchors(
            packet, topk=topk, text_anchor_policy=text_anchor_policy
        )
        posterior = self.gaussian_posterior(
            rows, weights, scale=scale, temperature=temperature,
            membership_margin=membership_margin,
            posterior_chunk_size=posterior_chunk_size,
        )
        if identity_extent_weight:
            if identity_extent_weight < 0:
                raise ValueError("identity extent weight differs")
            finite = identity[torch.isfinite(identity)]
            if not finite.numel():
                raise ValueError("identity extent score has no finite rows")
            quantiles = torch.quantile(finite, torch.tensor([0.1, 0.5, 0.9], device=finite.device))
            spread = (quantiles[2] - quantiles[0]).clamp_min(1e-4)
            identity_logit = (identity - quantiles[1]) / spread
            base_logit = torch.logit(posterior.clamp(1e-5, 1 - 1e-5))
            posterior = torch.sigmoid(
                base_logit + float(identity_extent_weight) * identity_logit
            )
        return posterior, identity

    @torch.no_grad()
    def calibrated_posterior_from_packet(
        self,
        packet: QueryPacket,
        calibrator: NullCalibratedPosterior,
        *,
        scale: float,
        topk: int = 8,
        temperature: float = 0.15,
        posterior_chunk_size: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse disentangled text evidence into one reusable Gaussian posterior."""

        identity, null, unknown = self.semantic_text_evidence(packet)
        instance, returned_identity = self.posterior_from_packet(
            packet,
            scale=scale,
            topk=topk,
            temperature=temperature,
            posterior_chunk_size=posterior_chunk_size,
            text_anchor_policy="positive",
        )
        if not torch.equal(identity, returned_identity):
            raise RuntimeError("positive identity evidence changed during expansion")
        posterior = calibrator(
            identity=identity,
            instance=instance,
            null=null,
            negative=torch.sigmoid((null - identity) * self.text_logit_scale),
            unknown=unknown,
            boundary=self.boundary_probability(),
            reliability=self.reliability,
        )
        return posterior, identity, instance

    @torch.no_grad()
    def gaussian_posterior(
        self,
        support_rows: torch.Tensor,
        support_weights: torch.Tensor,
        *,
        scale: float,
        temperature: float = 0.15,
        membership_margin: float = 0.0,
        posterior_chunk_size: int = 0,
    ) -> torch.Tensor:
        rows = torch.as_tensor(
            support_rows, device=self.model.memory.device, dtype=torch.long
        ).reshape(-1)
        weights = torch.as_tensor(
            support_weights, device=self.model.memory.device
        ).float().reshape(-1)
        if rows.shape != weights.shape or not rows.numel():
            raise ValueError("authorized query support axes differ")
        support = self.model.instance_view(scale, rows)
        prototype = pool_prototype(support, weights)
        if posterior_chunk_size < 0:
            raise ValueError("posterior chunk size differs")
        if posterior_chunk_size:
            values = []
            for start in range(0, self.model.memory.shape[0], posterior_chunk_size):
                stop = min(start + posterior_chunk_size, self.model.memory.shape[0])
                chunk_rows = torch.arange(start, stop, device=self.model.memory.device)
                values.append(membership_from_prototype(
                    self.model.instance_view(scale, chunk_rows), prototype,
                    temperature=temperature, margin=membership_margin,
                ))
            return torch.cat(values)
        return membership_from_prototype(
            self.model.instance_view(scale), prototype, temperature=temperature,
            margin=membership_margin,
        )

    @torch.no_grad()
    def boundary_probability(self) -> torch.Tensor:
        return self.boundary_head(self.model.boundary_view()).squeeze(-1).sigmoid()

    @staticmethod
    def render_posterior(
        posterior: torch.Tensor,
        gaussian_ids: torch.Tensor,
        pixel_ids: torch.Tensor,
        contribution_weights: torch.Tensor,
        *,
        num_pixels: int,
    ) -> torch.Tensor:
        """Render the exact posterior object returned by ``gaussian_posterior``."""

        return render_membership(
            posterior, gaussian_ids, pixel_ids, contribution_weights,
            num_pixels=num_pixels,
        )


def load_query_interface(
    scene_state_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    identity_adapter_path: str | Path | None = None,
    text_alignment_path: str | Path | None = None,
    text_negative_path: str | Path | None = None,
    text_logit_scale: float = 10.0,
    center_semantic_identity: bool = False,
) -> StructuredGaussianQueryInterface:
    path = Path(scene_state_path).resolve(strict=True)
    payload = torch.load(path, map_location="cpu")
    if payload.get("schema") != "radio_gs.sugm_v3.unknown_aware_scene_state.v1":
        raise ValueError("query input is not a sealed SUGM-v3 scene state")
    metadata = payload.get("metadata", {})
    if (
        metadata.get("persistent_gaussian_state") != "exactly_one_d512_plus_five_scalars"
        or metadata.get("gaussian_indexed_high_dimensional_sidecars") != 0
        or not metadata.get("source_only")
        or metadata.get("target_rgb_opened")
        or metadata.get("benchmark_metrics_opened")
    ):
        raise ValueError("scene state violates the canonical query contract")
    membership = metadata.get("inputs", {}).get("membership", {})
    authority_hash = membership.get("sha256")
    latent = torch.as_tensor(payload["latent"]).float()
    reliability = torch.as_tensor(payload["reliability"]).float()
    validate_scene_state(latent, reliability, source_authority_sha256=authority_hash)
    if Path(membership.get("path", "")).resolve(strict=True) and sha256_file(
        Path(membership["path"]).resolve(strict=True)
    ) != authority_hash:
        raise ValueError("scene-state source authority hash differs on disk")
    global_state = payload.get("global_state_dict", {})
    if any(key.startswith("_owned_training_blocks.") for key in global_state):
        raise ValueError("training-only Gaussian buffers leaked into deployment")
    layout = SharedPrivateLayout()
    model = LowRankPrivateBranchMemory(latent, layout=layout)
    current = model.state_dict()
    model_state = {"memory": latent}
    for key in current:
        if key == "memory":
            continue
        if key not in global_state or torch.as_tensor(global_state[key]).shape != current[key].shape:
            raise ValueError(f"deployment global state lacks {key}")
        model_state[key] = torch.as_tensor(global_state[key])
    model.load_state_dict(model_state, strict=True)
    boundary_head = nn.Linear(layout.boundary, 1)
    boundary_head.load_state_dict({
        "weight": torch.as_tensor(global_state["boundary_head.weight"]),
        "bias": torch.as_tensor(global_state["boundary_head.bias"]),
    })
    identity_adapter = None
    if identity_adapter_path is not None:
        adapter_path = Path(identity_adapter_path).resolve(strict=True)
        adapter_payload = torch.load(adapter_path, map_location="cpu")
        if adapter_payload.get("schema") != "radio_gs.sugm_v3.cross_view_identity_adapter.v1":
            raise ValueError("identity adapter schema differs")
        adapter_metadata = adapter_payload.get("metadata", {})
        if (
            not adapter_metadata.get("source_only")
            or adapter_metadata.get("dev_residue_opened")
            or adapter_metadata.get("audit_residue_opened")
            or adapter_metadata.get("benchmark_metrics_opened")
            or adapter_metadata.get("gaussian_indexed_state_added") != 0
        ):
            raise ValueError("identity adapter selection authority differs")
        identity_adapter = LowRankIdentityAdapter(
            dimension=int(adapter_payload["dimension"]), rank=int(adapter_payload["rank"])
        )
        identity_adapter.load_state_dict(adapter_payload["state_dict"], strict=True)
    text_alignment = None
    if text_alignment_path is not None:
        alignment_path = Path(text_alignment_path).resolve(strict=True)
        alignment_payload = torch.load(alignment_path, map_location="cpu")
        alignment_schema = alignment_payload.get("schema")
        if alignment_schema not in (
            "radio_gs.sugm_v3.orthogonal_text_alignment.v1",
            "radio_gs.sugm_v3.affine_text_alignment.v1",
            "radio_gs.sugm_v3.direct_text_projection.v1",
        ):
            raise ValueError("text alignment schema differs")
        alignment_metadata = alignment_payload.get("metadata", {})
        if (
            not alignment_metadata.get("source_only")
            or alignment_metadata.get("benchmark_metrics_opened")
            or alignment_metadata.get("gaussian_indexed_state_added") != 0
        ):
            raise ValueError("text alignment selection authority differs")
        if alignment_schema == "radio_gs.sugm_v3.orthogonal_text_alignment.v1":
            text_alignment = OrthogonalTextAlignment(alignment_payload["matrix"])
        elif alignment_schema == "radio_gs.sugm_v3.affine_text_alignment.v1":
            text_alignment = AffineTextAlignment(
                alignment_payload["matrix"], alignment_payload["bias"]
            )
        else:
            text_alignment = DirectTextProjection(alignment_payload["basis"])
    text_negative_tokens = None
    if text_negative_path is not None:
        negative_path = Path(text_negative_path).resolve(strict=True)
        negative_payload = torch.load(negative_path, map_location="cpu")
        if [str(value).casefold() for value in negative_payload.get("queries", [])] != [
            "object", "things", "stuff", "texture"
        ]:
            raise ValueError("canonical text negative vocabulary differs")
        text_negative_tokens = torch.as_tensor(negative_payload["embeddings"]).float()
    interface = StructuredGaussianQueryInterface(
        model,
        reliability,
        boundary_head,
        siglip_mean=global_state.get("codec.siglip_mean"),
        siglip_basis=global_state.get("codec.siglip_basis"),
        identity_adapter=identity_adapter,
        text_alignment=text_alignment,
        text_negative_tokens=text_negative_tokens,
        text_logit_scale=text_logit_scale,
        center_semantic_identity=center_semantic_identity,
    )
    return interface.to(device).eval()


__all__ = ["StructuredGaussianQueryInterface", "load_query_interface"]
