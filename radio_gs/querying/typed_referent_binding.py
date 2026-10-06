"""Separate lexical semantic evidence from conditional instance relations.

The appearance model can only mix native query descriptors. It has no scene
geometry or visual projection. Its parameters are distinct from the relation
model, allowing semantic supervision to be frozen before instance learning.
"""
import torch
from torch import nn


class LexicalIdentity(nn.Module):
    def __init__(self, text_width=1152, width=64, native_scale=1.):
        super().__init__()
        self.native_scale = native_scale
        self.query = nn.Sequential(nn.LayerNorm(text_width), nn.Linear(text_width, width), nn.GELU())
        self.channel = nn.Sequential(nn.LayerNorm(text_width), nn.Linear(text_width, width), nn.GELU())
        self.weight = nn.Sequential(nn.Linear(width * 2 + 1, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.weight[-1].weight);nn.init.zeros_(self.weight[-1].bias)

    def forward(self, query, return_weights=False):
        target = query['roles'] == 1
        states = torch.cat((query['query_tokens'][None], query['phrase_tokens'][target]))
        scores = torch.cat((query['native'][None], query['phrase_scores'][target]))
        is_full = states.new_zeros((len(states), 1));is_full[0] = 1
        feature = torch.cat((self.query(query['query_tokens']).expand(len(states), -1), self.channel(states), is_full), -1)
        weights = self.weight(feature).squeeze(-1).float().softmax(0)
        with torch.autocast(device_type=scores.device.type, enabled=False):
            identity = weights @ scores.float() * self.native_scale
        return (identity, weights) if return_weights else identity


class ConditionalRelations(nn.Module):
    def __init__(self, text_width=1152, width=64):
        super().__init__()
        self.language = nn.Sequential(nn.LayerNorm(text_width), nn.Linear(text_width, width), nn.GELU())
        self.energy = nn.Sequential(nn.Linear(width * 2 + 9, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.energy[-1].weight);nn.init.zeros_(self.energy[-1].bias)

    def forward(self, query, witness='native'):
        landmark = query['roles'] == 2
        if not landmark.any():
            return torch.zeros_like(query['native'])
        full = self.language(query['query_tokens'])
        phrases = self.language(query['phrase_tokens'][landmark])
        observed = query['spatial'][witness][landmark]
        background = query['spatial']['uniform'][landmark]
        terms = []
        for context, spatial, neutral in zip(phrases, observed, background):
            language = torch.cat((full, context)).expand(len(spatial), -1)
            observed_energy = self.energy(torch.cat((language, spatial.float()), -1)).squeeze(-1)
            neutral_energy = self.energy(torch.cat((language, neutral.float()), -1)).squeeze(-1)
            terms.append((observed_energy - neutral_energy).float())
        return torch.stack(terms).mean(0)


class TypedReferentBinding(nn.Module):
    def __init__(self, text_width=1152, width=64, native_scale=1.):
        super().__init__()
        self.identity = LexicalIdentity(text_width, width, native_scale)
        self.relation = ConditionalRelations(text_width, width)

    def forward(self, query, witness='native', return_components=False):
        identity = self.identity(query)
        relation = self.relation(query, witness)
        scores = identity + relation
        return (scores, identity, relation) if return_components else scores
