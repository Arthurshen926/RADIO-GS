"""Relative-only control of the existing complete-response value mechanism.

Discard query-shared context from the original [candidate, mean, difference]
input. This removes common-offset shortcuts, but may also discard useful query
difficulty information; performance must be measured with the same guard.
"""
import numpy as np


class CenteredResponseValue:
    def __init__(self,estimator):
        self.estimator=estimator

    def fit(self,x,y,*,sample_weight=None):
        x=np.asarray(x)
        if x.ndim!=2 or x.shape[1]%3 or not np.isfinite(x).all():
            raise ValueError('finite original set-conditioned features required')
        self.n_features_in_=x.shape[1]
        self.estimator.fit(x[:,2*x.shape[1]//3:],y,sample_weight=sample_weight)
        return self

    def predict(self,x):
        x=np.asarray(x)
        if x.ndim!=2 or x.shape[1]!=self.n_features_in_ or not np.isfinite(x).all():
            raise ValueError('prediction signature schema differs from fit')
        return self.estimator.predict(x[:,2*x.shape[1]//3:])

    @property
    def feature_importances_(self):
        # Keep audit tools compatible, explicitly recording discarded blocks.
        return np.r_[np.zeros(2*self.n_features_in_//3),self.estimator.feature_importances_]
