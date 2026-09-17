"""Keep geometry visibility distinct from evidence that trained a feature."""
import torch


def supervised_feature_rows(gradient, visible):
    if gradient.ndim != 2 or visible.ndim != 1 or len(gradient) != len(visible):
        raise ValueError('feature gradient and visibility axes differ')
    if not torch.isfinite(gradient).all():
        raise ValueError('nonfinite feature gradient')
    updated = gradient.ne(0).any(dim=1)
    if (updated & ~visible.bool()).any():
        raise ValueError('semantic gradient reached geometrically unobserved rows')
    return updated


class CoveragePixelSampler:
    """Visit each view's valid pixels before repeating, with a private RNG."""
    def __init__(self, valid_pixels, seed=1701):
        self.valid_pixels = valid_pixels
        self.seed = seed
        self.epochs = [0] * len(valid_pixels)
        self.offsets = [0] * len(valid_pixels)
        self.orders = [None] * len(valid_pixels)
        self.covered = [torch.zeros(len(v), dtype=torch.bool, device=v.device) for v in valid_pixels]

    def sample(self, view, count):
        if count < 1 or not 0 <= view < len(self.valid_pixels):
            raise ValueError('invalid pixel sampling request')
        pixels = self.valid_pixels[view]
        if not len(pixels):
            return pixels
        if self.orders[view] is None or self.offsets[view] == len(pixels):
            generator = torch.Generator().manual_seed(self.seed + view * 100003 + self.epochs[view])
            self.orders[view] = torch.randperm(len(pixels), generator=generator).to(pixels.device)
            self.offsets[view] = 0
            self.epochs[view] += 1
        start = self.offsets[view]
        indices = self.orders[view][start:start + count]
        self.offsets[view] += len(indices)
        self.covered[view][indices] = True
        return pixels[indices]
