"""Native seed-region completion without the supervised region quality model.

Experimental development candidate, selected after the frozen five-arm study.
This only completes the existing field-derived set; it is not full instance
scope or a promoted five-task system. No conditional forest/appearance reader.
"""
import numpy as np
import torch
from .scope_confidence import native_text_evidence,native_scope_confidence
from .joint_exclusion import JointExclusionDecoder
from .scope_readout import ScopeReadout


class SeedScopeDecoder:
    def __init__(self,field):
        if field.region_carrier is None:raise ValueError('field has no binary region ownership')
        self.field=field
        self.membership=field.region_carrier.matrix('membership').tocsc()
        self.area=np.asarray(self.membership.sum(0)).ravel()

    def select(self,probability):
        if probability.shape!=(len(self.area),) or not bool(torch.isfinite(probability).all()):raise ValueError('unaligned region response')
        values=probability.detach().cpu().numpy()
        eligible=np.flatnonzero((self.area>0)&(values>.5))
        if not len(eligible):return None
        return int(eligible[np.argmax(values[eligible])])


@torch.no_grad()
def query_text_seed_scope(field,query,negatives,*,seed_decoder=None,exclusion_decoder=None,native_reader=None):
    decoder=seed_decoder or SeedScopeDecoder(field)
    exclusion=exclusion_decoder or JointExclusionDecoder(field)
    if decoder.field is not field or exclusion.field is not field or (native_reader is not None and native_reader.field is not field):
        raise ValueError('decoder bound to another field')
    initial,probability=native_text_evidence(field,query,negatives) if native_reader is None else native_reader(query,negatives)
    accepted,_=native_scope_confidence(field,initial,probability)
    member,details=exclusion.decode(initial,accepted,joint=False)
    before=member>.5;seed=decoder.select(probability)
    if seed is not None:
        rows=decoder.membership[:,seed].indices
        member[torch.from_numpy(rows.astype(np.int64)).to(member.device)]=1.
    member=member.masked_fill(~(field.observed|field.semantic_measure.supported),0)
    return ScopeReadout(initial,member,member>.5,accepted,dict(details,
        seed_region=seed,added=int(((member>.5)&~before).sum()),addition_supported=True,
        conditional_quality_model_used=False,region_appearance_features_used=False,
        additional_3d_label_model_used=False,source_access=False,query_parameter_fitting=False,
        experimental=True,development_candidate=True,default_promoted=False))
