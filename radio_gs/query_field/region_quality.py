"""Tensor-only v10 conditional region quality function.

This transfers the learned three-output function, not the meaning of its input
features. Mesh regional membership and Gaussian projected mass are different
measures: this module is NOT enabled in CapabilityField until that interface
has independent validation. No scene, query, source loader or estimator is kept.
"""
import math
import torch
from torch import nn

TARGETS = ('candidate_purity', 'candidate_visible_recall', 'added_element_purity')


def compile_region_quality(checkpoint):
    """Offline only. Preserve float32 inputs and double split/leaf semantics."""
    from radio_gs.querying.conditional_region_membership import FEATURE_NAMES
    model = checkpoint['estimator']
    if tuple(checkpoint['target_order']) != TARGETS or checkpoint['feature_names'] != FEATURE_NAMES:
        raise ValueError('conditional quality feature/target semantics differ')
    if model.n_outputs_ != 3 or model.n_features_in_ != len(FEATURE_NAMES):
        raise ValueError('conditional quality dimensions differ')
    arrays = {k: [] for k in ['roots', 'feature', 'left', 'right', 'cut', 'value']}
    offset = 0
    for estimator in model.estimators_:
        t = estimator.tree_
        arrays['roots'].append(offset)
        for i in range(t.node_count):
            arrays['feature'].append(max(int(t.feature[i]), 0))
            for key, children in [('left', t.children_left), ('right', t.children_right)]:
                arrays[key].append(offset + (i if children[i] < 0 else int(children[i])))
            arrays['cut'].append(float(t.threshold[i]))
            arrays['value'].append(t.value[i, :, 0].tolist())
        offset += t.node_count
    state = {k: torch.tensor(v, dtype=torch.float64 if k in ['cut', 'value'] else torch.long)
             for k, v in arrays.items()}
    state.update(depth=torch.tensor(max(t.tree_.max_depth for t in model.estimators_)),
                 dimensions=torch.tensor(model.n_features_in_),
                 addition_threshold=torch.tensor(checkpoint['addition_threshold'], dtype=torch.float64))
    return state


class RegionQuality(nn.Module):
    def __init__(self, state):
        super().__init__()
        required = {'roots', 'feature', 'left', 'right', 'cut', 'value', 'depth', 'dimensions', 'addition_threshold'}
        if set(state) != required:
            raise ValueError('incomplete conditional quality state')
        for key, value in state.items():
            self.register_buffer(key, value.detach().clone())
        n = len(self.feature)
        for key in ['depth', 'dimensions', 'addition_threshold']:
            if getattr(self, key).ndim != 0:
                raise ValueError('invalid scalar quality metadata')
        if n == 0 or self.roots.ndim != 1 or not len(self.roots) or self.value.shape != (n, 3):
            raise ValueError('invalid quality arrays')
        if any(getattr(self, k).shape != (n,) for k in ['feature', 'left', 'right', 'cut']):
            raise ValueError('unaligned quality arrays')
        for key in ['roots', 'feature', 'left', 'right', 'depth', 'dimensions']:
            if getattr(self, key).dtype != torch.long:
                raise ValueError('quality addresses must be int64')
        if self.cut.dtype != torch.float64 or self.value.dtype != torch.float64 or self.addition_threshold.dtype != torch.float64:
            raise ValueError('quality split, leaf and admission precision must be preserved')
        if int(self.dimensions) != 16 or not 0 <= int(self.depth) < n:
            raise ValueError('invalid quality dimensions/depth')
        if not math.isfinite(float(self.addition_threshold)) or not 0 <= float(self.addition_threshold) <= 1.000001:
            raise ValueError('invalid quality admission threshold')
        for key in ['roots', 'left', 'right']:
            if bool(((getattr(self, key) < 0) | (getattr(self, key) >= n)).any()):
                raise ValueError('invalid quality node address')
        if bool(((self.feature < 0) | (self.feature >= int(self.dimensions))).any()):
            raise ValueError('invalid quality feature address')
        if not bool(torch.isfinite(self.cut).all() & torch.isfinite(self.value).all()):
            raise ValueError('nonfinite quality state')
        ids = torch.arange(n, device=self.left.device)
        leaf = self.left == ids
        if not torch.equal(leaf, self.right == ids):
            raise ValueError('invalid quality leaf')
        if bool(((~leaf) & ((self.left <= ids) | (self.right <= ids))).any()):
            raise ValueError('quality tree must have forward child addresses')

    def forward(self, features, chunk_size=4096):
        if features.ndim != 2 or features.shape[1] != int(self.dimensions) or chunk_size <= 0:
            raise ValueError('invalid quality features')
        x32 = features.float()
        if not bool(torch.isfinite(x32).all()):
            raise ValueError('nonfinite quality input')
        result = []
        for x in x32.split(chunk_size):
            nodes = self.roots.expand(len(x), -1)
            for _ in range(int(self.depth)):
                values = x.gather(1, self.feature[nodes]).double()
                nodes = torch.where(values <= self.cut[nodes], self.left[nodes], self.right[nodes])
            if not bool(((self.left[nodes] == nodes) & (self.right[nodes] == nodes)).all()):
                raise ValueError('quality depth does not reach leaves')
            # Match canonical sklearn n_jobs=1 accumulation, including ties at
            # the calibrated threshold. A float32 output or tree-reduced mean
            # can silently alter >= admission at a calibration boundary.
            prediction = torch.zeros((len(x), 3), dtype=torch.float64, device=x.device)
            for tree in nodes.unbind(1):
                prediction += self.value[tree]
            result.append((prediction / len(self.roots)).clamp(0, 1))
        return torch.cat(result) if result else self.value.new_empty((0, 3))

    def eligible(self, prediction, additions, same_view):
        """Exact v10 extension/no-extension eligibility, before per-view merit."""
        if prediction.ndim != 2 or prediction.shape[1] != 3 or additions.shape != prediction.shape[:1] or same_view.shape != additions.shape or same_view.dtype != torch.bool:
            raise ValueError('unaligned quality admission')
        if not bool(torch.isfinite(prediction).all() & torch.isfinite(additions).all()) or bool((additions < 0).any()):
            raise ValueError('invalid quality admission values')
        extension = additions > 0
        accepted = torch.where(extension, prediction[:, 2], prediction[:, 0]) >= self.addition_threshold
        return accepted & (~same_view | extension)
