"""Query-only mention extraction and scene binding in frozen teacher coordinates.

Language states identify a phrase's role. They never replace the independently
pooled teacher descriptor used to recognize that phrase in the scene.
"""
import re
import numpy as np
import torch
from torch import nn


class MentionRoles(nn.Module):
    def __init__(self, text_width=1152, width=128):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(text_width), nn.Linear(text_width, width),
                                 nn.GELU(), nn.Linear(width, 5))

    def forward(self, tokens):
        # O, B-target, I-target, B-landmark, I-landmark.
        return self.net(tokens)


def decode_mentions(expression, offsets, logits):
    """Decode all predicted contiguous spans; no known entity boundaries/IDs."""
    offsets = np.asarray(offsets)
    logits = np.asarray(logits)
    spans = []
    active = None
    # Decode complete original words, not fragments such as the final subword
    # of "stove". Tokenizer offsets may include preceding spaces.
    for word in re.finditer(r"\w+(?:['’\-]\w+)*|[^\w\s]", expression):
        start, end = word.span()
        ids = np.flatnonzero((offsets[:, 1] > start) & (offsets[:, 0] < end))
        if not len(ids):
            active = None
            continue
        label = int(logits[ids].mean(0).argmax())
        role = 1 if label in (1, 2) else 2 if label in (3, 4) else 0
        if role == 0:
            active = None
            continue
        begins = label in (1, 3)
        if active is None or active['role'] != role or begins:
            active = dict(start=int(start), end=int(end), role=role, tokens=ids.tolist())
            spans.append(active)
        else:
            active['end'] = int(end)
            active['tokens'] = sorted(set(active['tokens']) | set(ids.tolist()))
    for span in spans:
        span['phrase'] = expression[span['start']:span['end']]
    return spans


class NativePhraseBinding(nn.Module):
    """Predict a residual from native identity and explicit scene witnesses.

Every supported region remains a candidate and a possible landmark. The same
network supports a matched control with query-independent scene witnesses.
"""
    def __init__(self, width=64, text_width=1152, native_scale=1.):
        super().__init__()
        self.native_scale = native_scale
        self.language = nn.Sequential(nn.LayerNorm(text_width), nn.Linear(text_width, width), nn.GELU())
        self.local = nn.Sequential(nn.Linear(width + 10, width), nn.GELU(), nn.Linear(width, 1))
        # Region geometry, marginalized spatial relations, three untouched
        # native similarities, and contextual language.
        self.relation = nn.Sequential(nn.Linear(width * 2 + 20, width), nn.GELU(), nn.Linear(width, 1))
        for head in (self.local, self.relation):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, geometry, native, phrase_scores, phrase_tokens, roles, query_tokens, spatial,
                witness='native', return_components=False):
        # All similarities come from the frozen, independently pooled encoder.
        language = self.language(query_tokens)
        target = roles == 1
        target_score = phrase_scores[target].amax(0) if target.any() else native
        g = geometry
        local = torch.cat((g, native[:, None], target_score[:, None],
                           language.expand(len(g), -1)), dim=-1)
        local_correction = self.local(local).squeeze(-1)
        relation_correction = torch.zeros_like(local_correction)
        landmark = roles == 2
        if landmark.any():
            scores = phrase_scores[landmark].float()
            context = self.language(phrase_tokens[landmark])
            relations = spatial[witness][landmark]
            parts = []
            for i in range(len(context)):
                features = torch.cat((g, relations[i].float(),
                    native[:, None], target_score[:, None], scores[i, :, None],
                    language.expand(len(g), -1), context[i].expand(len(g), -1)), dim=-1)
                parts.append(self.relation(features).squeeze(-1))
            relation_correction = torch.stack(parts).mean(0)
        score = native.float() * self.native_scale + (local_correction + relation_correction).float()
        return (score, relation_correction) if return_components else score


@torch.inference_mode()
def spatial_witnesses(geometry, phrase_scores, native_scale, chunk=256):
    """Marginalize relations over every possible landmark, never its centroid.

The nine fixed basis functions span proximity at three room-relative scales
and signed axis relations. They are input features, not selection thresholds.
Chunking changes memory use only; every supported region contributes.
"""
    xyz = geometry[:, :3].float()
    scores = phrase_scores.float() * native_scale
    masses = {'native': scores.softmax(-1), 'reversed': (-scores).softmax(-1)}
    result = {key: [] for key in ['native', 'reversed', 'uniform']}
    for start in range(0, len(xyz), chunk):
        delta = xyz[start:start + chunk, None] - xyz[None]
        squared = delta.square().sum(-1)
        kernels = [torch.exp(-squared / (2 * scale * scale)) for scale in [.05, .15, .5]]
        for axis in range(3):
            positive = torch.sigmoid(delta[..., axis] / .05)
            kernels.extend([positive, 1 - positive])
        basis = torch.stack(kernels, dim=-1)
        for key, mass in masses.items():
            value = torch.einsum('pr,crk->pck', mass, basis)
            result[key].append(value.half())
        result['uniform'].append(basis.mean(1).half()[None])
    output = {key: torch.cat(parts, dim=1) for key, parts in result.items()}
    # A view shares one scene-only control tensor across all phrases.
    output['uniform'] = output['uniform'].expand(len(scores), -1, -1)
    return output
