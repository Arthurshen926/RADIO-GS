"""Query-independent, bounded native decoder correction; no second element map."""
import math
import torch
from torch.nn import functional as F
from radio_gs.models.native_decoder_readout import affine_decoder_cosine


def _affine_norm(z, weight, bias):
    quadratic = ((z @ (weight.T @ weight)) * z).sum(-1)
    cross = 2 * (z @ (weight.T @ bias))
    constant = bias.square().sum()
    squared = quadratic + cross + constant
    near = squared <= 32 * torch.finfo(z.dtype).eps * (quadratic.abs() + cross.abs() + constant.abs())
    norm = squared.clamp_min(1e-24).sqrt()
    if near.any():
        ids = torch.where(near)[0]
        exact = torch.cat([F.linear(z[rows], weight, bias).norm(dim=-1) for rows in ids.split(4096)])
        norm = norm.index_copy(0, ids, exact)
    return norm


def bounded_native_cosine(z, query, base_weight, base_bias, residual_weight, residual_bias, relative_bound):
    """Cosine of base + clipped residual in the *original* query space.

    The correction has norm <= relative_bound * norm(base). The base is not
    rewritten. This protects an angular neighbourhood, not every query rank.
    Augmented compact coordinates avoid a persistent N x native_dim tensor.
    """
    if not math.isfinite(relative_bound) or not 0 <= relative_bound <= .5:
        raise ValueError('native residual relative bound must be in [0,.5]')
    if relative_bound == 0:
        return affine_decoder_cosine(z[:, :base_weight.shape[1]], query, base_weight, base_bias)
    if residual_weight.shape != (len(base_bias), z.shape[1]) or residual_bias.shape != base_bias.shape:
        raise ValueError('native residual axes differ')
    if any(not torch.isfinite(x).all() for x in (z, query, base_weight, base_bias, residual_weight, residual_bias)):
        raise ValueError('nonfinite native residual input')
    base_norm = _affine_norm(z[:, :base_weight.shape[1]], base_weight, base_bias)
    residual_norm = _affine_norm(z, residual_weight, residual_bias)
    gate = (relative_bound * base_norm / residual_norm.clamp_min(1e-12)).clamp_max(1)
    padded = F.pad(base_weight, (0, z.shape[1] - base_weight.shape[1]))
    compact = torch.cat([z, gate[:, None] * z, gate[:, None]], dim=-1)
    weight = torch.cat([padded, residual_weight, residual_bias[:, None]], dim=-1)
    return affine_decoder_cosine(compact, query, weight, base_bias)
