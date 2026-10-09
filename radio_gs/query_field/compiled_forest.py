"""Tensor-only execution of an independently fitted binary decision forest.

Compilation is offline. Runtime owns no estimator, joblib loader or source path.
Double thresholds and leaf values retain sklearn's float32-input semantics.
"""
import torch
from torch import nn


def compile_forest(estimator, threshold):
    import numpy as np
    if estimator.n_classes_ != 2 or not np.array_equal(estimator.classes_, [False, True]):
        raise ValueError('forest requires binary zero/one classes')
    roots, feature, left, right, cut, value = [], [], [], [], [], []
    offset = 0
    for model in estimator.estimators_:
        tree = model.tree_; ids = np.arange(tree.node_count) + offset
        roots.append(offset)
        feature.extend(np.maximum(tree.feature, 0))
        left.extend(np.where(tree.children_left < 0, ids, tree.children_left + offset))
        right.extend(np.where(tree.children_right < 0, ids, tree.children_right + offset))
        cut.extend(tree.threshold)
        counts = tree.value[:, 0]
        value.extend(counts[:, 1] / counts.sum(1))
        offset += tree.node_count
    return dict(roots=torch.tensor(roots), feature=torch.tensor(feature),
        left=torch.tensor(left), right=torch.tensor(right), cut=torch.tensor(cut, dtype=torch.float64),
        value=torch.tensor(value, dtype=torch.float64),
        depth=torch.tensor(max(t.tree_.max_depth for t in estimator.estimators_)),
        dimensions=torch.tensor(estimator.n_features_in_), threshold=torch.tensor(threshold, dtype=torch.float64))


class CompiledForest(nn.Module):
    def __init__(self, state):
        super().__init__()
        if set(state) != {'roots', 'feature', 'left', 'right', 'cut', 'value', 'depth', 'dimensions', 'threshold'}:
            raise ValueError('incomplete compiled forest')
        for key, value in state.items(): self.register_buffer(key, value.detach())
        n = len(self.feature)
        if not n or not len(self.roots) or any(getattr(self, k).shape != (n,) for k in ['left','right','cut','value']):
            raise ValueError('invalid forest arrays')
        if not 0 < float(self.threshold) < 1 or int(self.depth) < 0:
            raise ValueError('invalid forest decision')
        for key in ['roots','left','right']:
            if bool(((getattr(self,key) < 0) | (getattr(self,key) >= n)).any()):
                raise ValueError('invalid forest node address')
        if bool(((self.feature < 0) | (self.feature >= int(self.dimensions))).any()):
            raise ValueError('invalid forest feature address')
        if not torch.isfinite(self.cut).all() or not torch.isfinite(self.value).all():
            raise ValueError('nonfinite forest state')

    def probability(self, features, chunk_size=8192):
        if features.ndim != 2 or features.shape[1] != int(self.dimensions):
            raise ValueError('forest feature dimensions differ')
        values = []
        for x in features.float().split(chunk_size):
            nodes = self.roots.expand(len(x), -1)
            for _ in range(int(self.depth)):
                inputs = x.gather(1, self.feature[nodes]).double()
                nodes = torch.where(inputs <= self.cut[nodes], self.left[nodes], self.right[nodes])
            values.append(self.value[nodes].mean(1).float())
        return torch.cat(values) if values else features.new_empty(0)
