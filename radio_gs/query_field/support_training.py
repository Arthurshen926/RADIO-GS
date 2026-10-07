"""Source-only support episodes that use the deployed uncertain-ray adapter."""
import torch

from .adapters import raster_support
from .contracts import Support


class TrainingRays:
    """Keep the immutable sparse source rays on CPU for small support gathers."""
    def __init__(self, operator):
        operator = operator.detach().cpu().coalesce()
        self.pixels, self.rows = operator.indices()
        self.values = operator.values()
        self.count = operator.shape[1]

    def support(self, pixels, device):
        pixels = torch.as_tensor(pixels).long().cpu().unique(sorted=True)
        bounds = torch.searchsorted(self.pixels, torch.stack([pixels, pixels+1]))
        rows, values, local_pixels = [], [], []
        for index, (start, end) in enumerate(zip(*bounds.tolist())):
            rows.append(self.rows[start:end])
            values.append(self.values[start:end])
            local_pixels.append(torch.full((end-start,), index, dtype=torch.long))
        operator = torch.sparse_coo_tensor(torch.stack([torch.cat(local_pixels), torch.cat(rows)]),
                                           torch.cat(values), (len(pixels), self.count)).coalesce()
        support, _ = raster_support(operator, torch.ones(len(pixels)))
        return Support(support.indices.to(device), support.weights.to(device))


def sample_pixels(candidates, first, generator, count=4):
    """Use the original sampled positive plus distinct extra source interiors."""
    candidates = candidates.cpu()
    remaining = candidates[candidates != int(first)]
    order = torch.randperm(len(remaining), generator=generator)[:max(0, count-1)]
    return torch.cat([torch.tensor([int(first)]), remaining[order]])


def episode_support(policy, step, view, candidate_pixels, first_pixel, negative_pixels, generator, device):
    """Two matched policies differ only by signed exclusion on the fourth slot.

    One forward episode per step, no extra optimizer updates. Negative pixels
    are explicitly known source exclusions with observed dominant contributors.
    Missing reliable negatives retain the unsigned pooled support and are counted.
    """
    if policy not in {"single", "mixed_unsigned", "mixed_signed"}:
        raise ValueError("unknown source support policy")
    slot = 0 if policy == "single" else step % 4
    if slot == 0:
        anchor = view["best_element"][first_pixel].to(device)
        return Support(anchor.reshape(1), torch.ones(1, device=device)), None, "anchor"
    if slot == 1:
        return view["training_rays"].support([int(first_pixel)], device), None, "ray"
    pixels = sample_pixels(candidate_pixels, first_pixel, generator)
    positive = view["training_rays"].support(pixels, device)
    # Draw the same negative pool in both arms to preserve future support draws.
    if len(negative_pixels):
        first = negative_pixels[int(torch.randint(len(negative_pixels), (1,), generator=generator))]
        negative = sample_pixels(negative_pixels, first, generator)
    else:
        negative = None
    if slot == 3 and policy == "mixed_signed":
        if negative is None:
            return positive, None, "pooled_missing_negative"
        return positive, view["training_rays"].support(negative, device), "signed_pooled"
    return positive, None, "pooled"
