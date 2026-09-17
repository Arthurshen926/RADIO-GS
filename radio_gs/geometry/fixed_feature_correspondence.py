"""Source geometry correspondences with an optional frozen appearance veto."""
import torch
from torch.nn import functional as F


def fixed_feature_correspondences(views, features, unit, *, stride=4, minimum_cosine=.6, cycle_pixels=2.):
    if len(views) != len(features) or stride < 1 or not -1 <= minimum_cosine <= 1 or cycle_pixels <= 0:
        raise ValueError('invalid fixed-feature correspondence policy')
    banks = {'geometry': [], 'feature': []}
    audit = []
    def sample(feature, x, y, width, height):
        grid = torch.stack([2*(x+.5)/width-1, 2*(y+.5)/height-1], -1).reshape(1, 1, -1, 2)
        value = F.grid_sample(feature.float()[None], grid, mode='bilinear', align_corners=False)[0, :, 0].T
        return F.normalize(value, dim=1)
    for i, src in enumerate(views):
        h, w = src['depth'].shape
        mask = src['valid'].clone()
        yy, xx = torch.meshgrid(torch.arange(h, device=mask.device), torch.arange(w, device=mask.device), indexing='ij')
        mask &= (yy % stride == 0) & (xx % stride == 0)
        y, x = torch.where(mask)
        p = src['rays'][y, x] * src['depth'][y, x, None]
        world = p @ src['pose'][:3, :3].T + src['pose'][:3, 3]
        descriptor = sample(features[i], x, y, w, h)
        for j in range(i+1, min(len(views), i+3)):
            dst = views[j]
            dh, dw = dst['depth'].shape
            cam = (world-dst['pose'][:3, 3]) @ dst['pose'][:3, :3]
            uv = cam @ dst['k'].T
            dx, dy = (uv[:, 0]/uv[:, 2].clamp_min(1e-8)).floor().long(), (uv[:, 1]/uv[:, 2].clamp_min(1e-8)).floor().long()
            inside = (cam[:, 2] > 0) & (dx >= 0) & (dx < dw) & (dy >= 0) & (dy < dh)
            dx, dy = dx.clamp(0, dw-1), dy.clamp(0, dh-1)
            depth = dst['depth'][dy, dx]
            q = dst['rays'][dy, dx] * depth[:, None]
            back = ((q @ dst['pose'][:3, :3].T + dst['pose'][:3, 3])-src['pose'][:3, 3]) @ src['pose'][:3, :3]
            back_uv = back @ src['k'].T
            cycle = (back_uv[:, :2]/back_uv[:, 2:].clamp_min(1e-8)-torch.stack([x+.5, y+.5], 1)).norm(dim=1)
            n1 = src['normals'][y, x] @ src['pose'][:3, :3].T
            n2 = dst['normals'][dy, dx] @ dst['pose'][:3, :3].T
            accepted = inside & dst['valid'][dy, dx] & (back[:, 2] > 0) & (cycle <= cycle_pixels)
            accepted &= (cam[:, 2]-depth).abs() < .03*unit+.08*depth
            accepted &= (back[:, 2]-p[:, 2]).abs() < .02*unit+.03*p[:, 2]
            accepted &= F.cosine_similarity(n1, n2).abs() > .8
            cosine = (descriptor * sample(features[j], dx, dy, dw, dh)).sum(1)
            feature = accepted & (cosine >= minimum_cosine)
            audit.append(dict(source=src['frame'], target=dst['frame'], sampled=len(x), geometric=int(accepted.sum()), feature=int(feature.sum())))
            for name, keep in [('geometry', accepted), ('feature', feature)]:
                if int(keep.sum()) < 32:
                    continue
                count = int(keep.sum())
                banks[name].append((torch.full((count,), i, device=p.device), torch.full((count,), j, device=p.device),
                    p[keep] @ src['pose'][:3, :3].T, q[keep] @ dst['pose'][:3, :3].T,
                    (src['pose'][:3, 3]-dst['pose'][:3, 3]).expand(count, -1)))
    if not all(banks.values()):
        raise ValueError('insufficient corroborated mapping correspondences')
    return {name: tuple(torch.cat([row[k] for row in rows]) for k in range(5)) for name, rows in banks.items()}, audit
