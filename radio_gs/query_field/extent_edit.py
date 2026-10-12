"""Finite changes to a field-derived prior, without immutable predicted members.

Footprint penalizes the final set's size. Edit footprint instead penalizes
changing the supplied prior, with the same normalized physical measure and
coefficient. This is a diagnostic objective change, not a new identity model.
"""
import numpy as np
from .extent_decision_coupled import CoupledExtentDecision,coupled_decisions
from .extent_decision_linked import solve


class EditExtentDecision(CoupledExtentDecision):
    def solve(self,initial,target,positive,negative,confidence,cost,*,max_passes=4,
              decision_cutoff=.5,observation_weight=.5,regularizer='edit',proposal_domain=None):
        if regularizer not in {'edit','none'}:raise ValueError('unknown edit regularizer')
        if not np.isfinite(observation_weight) or not 0<=observation_weight<=1:raise ValueError('invalid observation weight')
        if proposal_domain is not None:
            proposal_domain=np.asarray(proposal_domain)
            if (proposal_domain.shape!=(self.count,) or proposal_domain.dtype!=bool
                    or (np.asarray(initial)&~proposal_domain).any()):
                raise ValueError('proposal domain must contain the initial feasible set')
        def kernel(*args):
            args=list(args);raw=args[12]
            # For binary S: |S-S0| = S0 + (1-2*S0)*S. Only the
            # internal unary coefficient is signed; physical footprint is not.
            offset=.1*np.dot(raw,args[7]) if regularizer=='edit' else 0.
            args[12]=np.where(args[7],-raw,raw) if regularizer=='edit' else np.zeros_like(raw)
            if proposal_domain is None:
                result=coupled_decisions(*args,self.optr,self.orows,self.oval,float(observation_weight))
            else:
                result=coupled_decisions(*args,self.optr,self.orows,self.oval,float(observation_weight),None,proposal_domain)
            selected,trace,flips,evaluated,converged=result
            return selected,trace+offset,flips,evaluated,converged
        selected,report=solve(self,initial,target,positive,negative,confidence,cost,
            max_passes=max_passes,decision_cutoff=decision_cutoff,_kernel=kernel)
        report.update(regularizer=regularizer,observation_weight=float(observation_weight),
            prior_members_locked=False,same_members_for_both_operators=True,
            proposal_domain_restricted=proposal_domain is not None,
            regularizer_coefficient=.1 if regularizer=='edit' else 0.)
        return selected,report
