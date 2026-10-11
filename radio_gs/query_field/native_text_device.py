"""Transient device mirror of native query functions, never query answers.

The discrete exclusion reference depends on float score ordering. Moving its
native input computation from CUDA to CPU can change ties and selected sets.
Mirror only necessary field buffers to preserve the established CUDA function
without moving unrelated coverage, local appearance or source evidence state.
"""
from types import SimpleNamespace
from .field import NativeCoordinates
from .semantic_measure import SemanticMeasure
from .scope_confidence import native_text_evidence


class NativeTextDevice:
    def __init__(self,field,device):
        self.field=field;self.device=device
        semantic=field.semantic
        self.native=SimpleNamespace(observed=field.observed.to(device),
            semantic=NativeCoordinates(semantic.codes.to(device),
                None if semantic.weight is None else semantic.weight.to(device),
                None if semantic.bias is None else semantic.bias.to(device)),
            semantic_measure=SemanticMeasure({k:v.to(device) for k,v in field.semantic_measure.state_dict().items()},len(field.xyz)),
            evidence=SimpleNamespace(region_semantic=field.evidence.region_semantic.to(device)))

    def __call__(self,query,negatives):
        prior,probability=native_text_evidence(self.native,query.to(self.device),negatives.to(self.device))
        return prior.to(self.field.xyz.device),probability.to(self.field.xyz.device)
