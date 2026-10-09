"""Shared conditional observation functions on Gaussian rows, not source assets.

Binary membership and joint visibility retain nested-mask disagreement. Native
moments preserve the *linear anchor-response function*, including its magnitude.
An optional compiled forest declares additional independent 3D supervision.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .compiled_forest import CompiledForest

KIND = 'joint_observation_function_v1'


class JointObservationEvidence(nn.Module):
    complete_carrier = True
    replaces_member_prior = True

    def __init__(self, state, count):
        super().__init__()
        forest = {k.removeprefix('forest.'): v for k, v in state.items() if k.startswith('forest.')}
        state = {k: v for k, v in state.items() if not k.startswith('forest.')}
        expected = {'views', 'number_of_views', 'native_moment', 'visible_fraction', 'native_supported'} | {
            f'{p}_{k}' for p in ['membership','visibility','association'] for k in ['pointer','index','value']}
        if set(state) != expected: raise ValueError('incomplete joint observation field')
        for key, value in state.items(): self.register_buffer(key, value.detach())
        self.count = count
        self.view_count = int(self.number_of_views)
        if self.view_count <= 0: raise ValueError('empty observation view domain')
        if self.native_moment.ndim != 2 or len(self.native_moment) != count:
            raise ValueError('unaligned native moment')
        if self.visible_fraction.shape != (count,) or self.native_supported.shape != (count,):
            raise ValueError('unaligned observation state')
        if not torch.isfinite(self.native_moment).all() or not torch.isfinite(self.visible_fraction).all():
            raise ValueError('nonfinite observation state')
        if bool(((self.views < 0) | (self.views >= self.view_count)).any()):
            raise ValueError('invalid source view groups')
        for prefix in ['membership','visibility','association']:
            pointer, index, value = [getattr(self, prefix+'_'+k) for k in ['pointer','index','value']]
            columns = self.view_count if prefix == 'visibility' else len(self.views)
            if pointer.shape != (count+1,) or int(pointer[0]) or int(pointer[-1]) != len(index) or len(value) != len(index):
                raise ValueError('unaligned observation CSR')
            if bool((pointer.diff() < 0).any()) or bool(((index < 0) | (index >= columns)).any()):
                raise ValueError('invalid observation CSR indices')
            if not torch.isfinite(value).all() or bool(((value <= 0) | (value > 1)).any()):
                raise ValueError('invalid observation coordinate')
            if prefix != 'association' and not bool((value == 1).all()):
                raise ValueError('joint visibility reference requires binary source observations')
        self.forest = CompiledForest(forest) if forest else None

    def matrix(self, prefix, rows=None):
        pointer, index, value = [getattr(self, prefix+'_'+k) for k in ['pointer','index','value']]
        if rows is not None:
            rows = rows.to(pointer.device); lengths = pointer[rows+1] - pointer[rows]
            offset = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
            positions = torch.arange(int(offset[-1]), device=rows.device) + torch.repeat_interleave(pointer[rows]-offset[:-1], lengths)
            pointer, index, value = offset, index[positions], value[positions]
        columns = self.view_count if prefix == 'visibility' else len(self.views)
        return torch.sparse_csr_tensor(pointer, index, value, size=(len(pointer)-1, columns), device=value.device)

    def joint_features(self, support, rows=None):
        support.validate(self.count)
        weights = support.weights.to(self.native_moment); weights = weights / weights.sum()
        coefficients = torch.sparse.mm(self.matrix('association', support.indices).transpose(0,1), weights[:,None]).flatten()
        confidence = coefficients.new_zeros(self.view_count).scatter_add_(0, self.views, coefficients)
        ids = torch.where(coefficients > 0)[0]
        association = torch.sparse_coo_tensor(torch.stack([ids,self.views[ids]]), coefficients[ids],
            (len(coefficients),self.view_count)).to_dense()
        votes = torch.sparse.mm(self.matrix('membership',rows), association)
        visible = self.matrix('visibility',rows).to_dense()
        known = visible * confidence[None,:]
        denominator = known.sum(1)
        safe = denominator.clamp_min(1e-12)
        any_mask = (known * (votes > 0)).sum(1) / safe
        all_masks = (known * ((votes/confidence[None,:].clamp_min(1e-12) >= 1-1e-6)
                             & (confidence[None,:] > 0))).sum(1) / safe
        return torch.stack([(votes.sum(1)/safe).clamp(0,1), any_mask, all_masks,
            any_mask-all_masks, denominator/confidence.sum().clamp_min(1e-12), (denominator>0).to(votes)],1)

    def features(self, field, support, rows=None):
        joint = self.joint_features(support, rows)
        ids = support.indices.to(field.xyz.device)
        weights = support.weights.to(field.xyz); weights = weights/weights.sum()
        xyz = field.xyz if rows is None else field.xyz[rows]
        centre = (field.xyz[ids]*weights[:,None]).sum(0)
        delta = (xyz-centre)/field.world_units_per_metre
        distance = delta.norm(dim=1); horizontal = delta[:,:2].norm(dim=1)
        moment = self.native_moment if rows is None else self.native_moment[rows]
        prototype = F.normalize((self.native_moment[ids]*weights[:,None]).sum(0),dim=0,eps=1e-8)
        visible = self.visible_fraction if rows is None else self.visible_fraction[rows]
        supported = self.native_supported if rows is None else self.native_supported[rows]
        return torch.cat([joint, (self.visible_fraction[ids]*weights).sum().expand(len(xyz),1),
            visible[:,None], delta/(1+delta.abs()), (distance/(1+distance))[:,None],
            (horizontal/(1+horizontal))[:,None], (moment@prototype)[:,None], supported[:,None].to(joint)],1)

    def member_logits(self, field, support, rows=None):
        if self.forest is None:
            probability = self.joint_features(support, rows)[:,0]
            threshold = probability.new_tensor(.5)
        else:
            probability = self.forest.probability(self.features(field,support,rows))
            threshold = self.forest.threshold.to(probability)
        # Preserve the independently calibrated strict decision through the
        # common .5 field interface; this is not a new probability calibration.
        return torch.logit(probability.clamp(1e-7,1-1e-7))-torch.logit(threshold)


def enable(field, state):
    if field.evidence is not None or field.world_units_per_metre is None:
        raise ValueError('joint observation function needs an unaugmented physical field')
    field.evidence = JointObservationEvidence(state,len(field.xyz)).to(field.xyz.device)
    field.evidence_kind = KIND
