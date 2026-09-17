"""A bounded structural correction, retaining the native query evidence."""
import math
import torch


def bounded_extent_logits(native_logits, extent_probability, overlap, strength):
    if not math.isfinite(strength) or not 0 <= strength <= 2:
        raise ValueError('structural strength must lie in [0,2]')
    if native_logits.shape != extent_probability.shape or overlap.shape != (native_logits.shape[1],):
        raise ValueError('aligned element/query probabilities and query overlaps required')
    if strength == 0:
        return native_logits
    return native_logits + strength * overlap.clamp(0, 1)[None] * (2*extent_probability.clamp(0, 1)-1)


def read_bounded_extent(field, query, modality, strength, rows=None):
    cosine = field.native_cosine(query,modality=modality)
    if modality in field.native_query_thresholds:
        native = field.readout_log_scale[modality].clamp(-4, 6).exp() * (cosine-field.native_query_thresholds[modality](query))
    else:
        native = field.cosine_logits(cosine, modality)
    if strength == 0:
        return native if rows is None else native[rows]
    head = field.grounded_transfer
    # Global selection must not depend on requested element rows or query batch.
    overlap = head.support_overlap(field.latent, native > 0, field.observed)
    quality, selected = overlap.max(1)
    features = head.features(field.latent if rows is None else field.latent[rows])
    probability = features @ head.decoder.weight[selected].T + head.decoder.bias[selected]
    if head.residual is not None:
        probability = probability + torch.nn.functional.linear(head.residual[1](head.residual[0](features)), head.residual[2].weight[selected], head.residual[2].bias[selected])
    return bounded_extent_logits(native if rows is None else native[rows], probability, quality, strength)
