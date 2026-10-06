"""Detector-conditioned native identity evidence, independent of query vocabulary.

Density groups describe score structure, not calibrated semantic correctness.
A unimodal/insufficient sample carries no relative identity veto. Higher native
score identifies the favored group only when the two-group model is supported.
"""
import numpy as np
from sklearn.mixture import GaussianMixture


def native_identity_posterior(scores, *, compare_unimodal=True):
    values=np.asarray(scores,dtype=np.float64)
    if len(values)<3 or values.std()<1e-6:
        return np.ones(len(values)),dict(model='insufficient_separation',samples=len(values))
    center=values.mean();scale=values.std();z=((values-center)/scale)[:,None]
    single=GaussianMixture(1,random_state=0).fit(z)
    mixture=GaussianMixture(2,random_state=0,n_init=5).fit(z)
    bic_single=float(single.bic(z));bic_mixture=float(mixture.bic(z))
    high=int(mixture.means_[:,0].argmax())
    two_probability=1/(1+np.exp(np.clip((bic_mixture-bic_single)/2,-700,700))) if compare_unimodal else 1.
    responsibility=mixture.predict_proba(z)[:,high]
    # The one-group hypothesis supplies no evidence for excluding an instance.
    posterior=(1-two_probability)+two_probability*responsibility
    return posterior,dict(model='one_vs_two_density_average' if compare_unimodal else 'two_density',samples=len(values),bic_single=bic_single,bic_mixture=bic_mixture,two_probability=float(two_probability),means=(mixture.means_[:,0]*scale+center).tolist(),weights=mixture.weights_.tolist())
