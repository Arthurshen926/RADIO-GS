"""Retain regional confidence as well as component/view confidence."""
from .mixture_paired_evidence import MixturePairedEvidence

KIND = 'confidence_mixture_partial_view_evidence_v1'


class ConfidenceMixtureEvidence(MixturePairedEvidence):
    def preference(self, positive, known):
        # p = known mass * conditional positive preference. Dividing out known
        # mass here would let a vanishing observation share a view equally with
        # a reliable one. The separate view confidence still bounds total mass.
        return positive


def enable(field, state):
    if field.evidence is not None or field.support_aggregation != 'normalized_prototype':
        raise ValueError('confidence mixture requires an unaugmented prototype core')
    field.evidence = ConfidenceMixtureEvidence(state, len(field.xyz)).to(field.xyz.device)
    field.evidence_kind = KIND
