"""Shared OpenCLIP text-readout helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F


NEGATIVE_PROMPTS = ("object", "things", "stuff", "texture")


def normalized_embeddings(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize rows while keeping all-zero rows zero."""
    values = x.float()
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("Cannot normalize non-finite OpenCLIP embeddings")
    norms = values.norm(dim=-1, keepdim=True)
    normalized = torch.where(
        norms > eps,
        values / norms.clamp_min(eps),
        torch.zeros_like(values),
    )
    if not bool(torch.isfinite(normalized).all()):
        raise FloatingPointError("OpenCLIP embedding normalization produced non-finite values")
    return normalized


def cosine_logits(features: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
    """Return normalized feature/text cosine logits."""
    visual = normalized_embeddings(features)
    text = normalized_embeddings(text_embeddings).to(device=visual.device, dtype=visual.dtype)
    return visual @ text.T


def build_prompt_variants(query: str, templates: Sequence[str]) -> list[str]:
    variants: list[str] = []
    for template in templates:
        if "{query}" in template:
            text = template.replace("{query}", query)
        elif "{}" in template:
            text = template.format(query)
        else:
            text = f"{template} {query}".strip()
        variants.append(text)
    return variants


def load_or_generate_openclip_prompt_ensemble_embeddings(
    queries: Sequence[str],
    device: torch.device,
    *,
    cache_path: str | Path | None = None,
    prompt_templates: Sequence[str] | None = None,
    model_name: str = "ViT-B-16",
    pretrained: str = "laion2b_s34b_b88k",
) -> torch.Tensor:
    """Encode OpenCLIP prompt ensembles and average one embedding per query."""
    templates = tuple(prompt_templates or ("{query}",))
    query_list = [str(query) for query in queries]
    cache = Path(cache_path) if cache_path else None
    if cache is not None and cache.exists():
        data = torch.load(cache, map_location="cpu")
        cached_queries = [str(q) for q in data.get("queries", [])]
        cache_compatible = (
            [str(t) for t in data.get("prompt_templates", ["{query}"])] == list(templates)
            and str(data.get("text_encoder", "openclip")) == "openclip"
            and str(data.get("openclip_model", data.get("model_name", model_name))) == model_name
            and str(data.get("openclip_pretrained", data.get("pretrained", pretrained))) == pretrained
        )
        if cache_compatible:
            bank = {query: embedding for query, embedding in zip(cached_queries, data["embeddings"])}
            missing = [query for query in query_list if query not in bank]
            if not missing:
                # A frozen all-scene bank is intentionally a superset of any
                # single scene.  Select rows without re-encoding or mutating
                # the declared cache on disk.
                selected = torch.stack([bank[query] for query in query_list])
                return normalized_embeddings(selected).to(device)

    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        precision="fp16",
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    flat_prompts: list[str] = []
    for query in query_list:
        flat_prompts.extend(build_prompt_variants(query, templates))
    with torch.inference_mode():
        tokens = torch.cat([tokenizer(prompt) for prompt in flat_prompts]).to(device)
        flat_emb = normalized_embeddings(model.encode_text(tokens))
        emb = flat_emb.reshape(len(query_list), len(templates), -1).mean(dim=1)
        emb = normalized_embeddings(emb)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "queries": query_list,
                "prompt_templates": list(templates),
                "text_encoder": "openclip",
                "openclip_model": model_name,
                "openclip_pretrained": pretrained,
                "model_name": model_name,
                "pretrained": pretrained,
                "embeddings": emb.detach().cpu(),
            },
            cache,
        )
    return emb


class OpenCLIPTextScorer:
    """OpenCLIP text encoder plus LERF relevance-map scoring."""

    def __init__(
        self,
        device: torch.device,
        *,
        model_name: str,
        pretrained: str,
        negative_prompts: Sequence[str] = NEGATIVE_PROMPTS,
    ) -> None:
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            precision="fp16",
        )
        model.eval()
        self.device = device
        self.model = model.to(device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.negative_prompts = tuple(negative_prompts)
        with torch.inference_mode():
            tokens = torch.cat([self.tokenizer(prompt) for prompt in self.negative_prompts]).to(device)
            self.neg_embeds = normalized_embeddings(self.model.encode_text(tokens))

    @torch.inference_mode()
    def _positive_embeds(self, prompts: Sequence[str]) -> torch.Tensor:
        tokens = torch.cat([self.tokenizer(prompt) for prompt in prompts]).to(self.device)
        return normalized_embeddings(self.model.encode_text(tokens))

    @torch.inference_mode()
    def relevance(self, sem_map: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        """Compute LangSplat-style positive-vs-negative relevance maps.

        Args:
            sem_map: Feature tensor shaped ``[levels,H,W,C]``.
            prompts: Positive text prompts, one per queried object.
        """
        pos_embeds = self._positive_embeds(prompts)
        phrase_embeds = torch.cat([pos_embeds, self.neg_embeds], dim=0).to(
            device=self.device,
            dtype=sem_map.dtype,
        )
        n_levels, height, width, channels = sem_map.shape
        n_prompts = len(prompts)
        n_negatives = len(self.negative_prompts)
        sem_flat = (
            sem_map.permute(0, 3, 1, 2)
            .reshape(n_levels, channels, -1)
            .permute(0, 2, 1)
            .contiguous()
        )
        sim = torch.einsum("nqc,pc->nqp", sem_flat, phrase_embeds)
        pos_vals = sim[:, :, :n_prompts]
        neg_vals = sim[:, :, n_prompts:]
        repeated_pos = pos_vals.unsqueeze(-1).repeat(1, 1, 1, n_negatives)
        repeated_neg = neg_vals.unsqueeze(2).repeat(1, 1, n_prompts, 1)
        sims = torch.stack([repeated_pos, repeated_neg], dim=-1)
        softmax = torch.softmax(10 * sims, dim=-1)
        min_pos_prob, _ = softmax[..., 0].min(dim=-1)
        return min_pos_prob.permute(0, 2, 1).reshape(n_levels, n_prompts, height, width)
