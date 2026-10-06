"""Read an explicitly reconstructed native field with the original query encoder."""
import torch
from torch.nn import functional as F


def attach_region_native_field(model, payload):
    features = payload['region_native_features'].to(model.latent.device)
    original = model.native_cosine

    def native_cosine(query, rows=None, *, modality='text'):
        if modality not in ('text', 'image'):
            return original(query, rows, modality=modality)
        query = F.normalize(query.float(), dim=-1)
        ids = torch.arange(len(features), device=features.device) if rows is None else rows
        values = []
        for chunk in ids.split(4096):
            value = F.normalize(features[chunk].float(), dim=-1)
            if model.native_residual_decoder is not None and modality in model.native_semantic_residual.get('modalities', ['text', 'image']):
                z = model.latent[chunk]
                base = model.native_decoder(z[:, :model.native_decoder.in_features])
                residual = model.native_residual_decoder(z)
                norm = base.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                gate = (model.native_semantic_residual['relative_bound'] * norm / residual.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp_max(1)
                if modality == 'image' and model.image_observed is not None:
                    gate = gate * model.observed[chunk, None]
                value = F.normalize(value + residual * gate / norm, dim=-1)
            values.append(value @ query.T)
        return torch.cat(values)

    model.native_cosine = native_cosine
    return model
