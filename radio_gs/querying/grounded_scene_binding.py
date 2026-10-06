"""Ground language mentions in map observations before resolving a referent.

All regions remain candidates. The language-to-scene attention has an explicit
training target, rather than expecting a final scalar loss to discover landmark
identity. Instance/phrase labels are never inputs to the deployed model.
"""
import torch
from torch import nn


class GroundedSceneBinding(nn.Module):
    def __init__(self, width=128, text_width=1152, native_width=1536, native_scale=1.):
        super().__init__()
        self.native_scale = native_scale
        self.visual = nn.Linear(native_width, width)
        self.language = nn.Linear(text_width, width)
        self.geometry = nn.Sequential(nn.Linear(8, width), nn.GELU(), nn.Linear(width, width))
        self.region_norm = nn.LayerNorm(width)
        self.mention_key = nn.Linear(width, width)
        self.region_key = nn.Linear(width, width)
        self.region_value = nn.Linear(width, width)
        self.grounded_norm = nn.LayerNorm(width)
        self.cross = nn.MultiheadAttention(width, 4, batch_first=True, dropout=.1)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(.1), nn.Linear(width * 2, width))
        self.output = nn.Linear(width, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, visual, geometry, tokens, valid, pooled, return_mentions=False,
                grounded_context=True):
        with torch.autocast(device_type=visual.device.type, enabled=False):
            native = pooled.float() @ visual.float().T * self.native_scale
        region = self.region_norm(self.visual(visual) + self.geometry(geometry))
        language = self.language(tokens)
        mentions = self.mention_key(language) @ self.region_key(region).T / region.shape[-1] ** .5
        if grounded_context:
            # Grounded values carry both actual scene appearance and position.
            # Padding is excluded by the downstream cross-attention mask.
            witnesses = mentions.softmax(-1) @ self.region_value(region)
        else:
            # Post-fit diagnostic only: remove query-specific landmark identity
            # while retaining the same scene and the original native query.
            witnesses = self.region_value(region).mean(0).expand_as(language)
        grounded = self.grounded_norm(language + witnesses)
        candidates = region[None].expand(len(tokens), -1, -1)
        context = self.cross(candidates, grounded, grounded, key_padding_mask=~valid, need_weights=False)[0]
        x = self.norm1(candidates + context)
        x = self.norm2(x + self.ff(x))
        scores = native + self.output(x).squeeze(-1)
        return (scores, mentions) if return_mentions else scores
