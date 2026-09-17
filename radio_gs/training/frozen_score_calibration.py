"""Fit a monotone scalar readout on source-only frozen cosine histograms."""
import torch
from torch.nn import functional as F


def fit_histogram_readout(positive, negative, *, balanced=False, score_min=-1., score_max=1.):
    if not torch.isfinite(torch.tensor([score_min,score_max])).all() or score_min>=score_max:
        raise ValueError('invalid histogram score interval')
    if positive.ndim != 1 or positive.shape != negative.shape or len(positive)<2:
        raise ValueError('matching score histograms required')
    if not torch.isfinite(positive).all() or not torch.isfinite(negative).all() or (positive<0).any() or (negative<0).any() or positive.sum()<=0 or negative.sum()<=0:
        raise ValueError('finite nonnegative evidence from both classes required')
    p,n=positive.double(),negative.double()
    if balanced:p,n=p/(2*p.sum()),n/(2*n.sum())
    else:
        mass=p.sum()+n.sum();p,n=p/mass,n/mass
    x=score_min+(torch.arange(len(p),device=p.device,dtype=p.dtype)+.5)*(score_max-score_min)/len(p)
    # Direct positive slope, projected to the deployment scale interval.
    scale=torch.tensor(10.,device=p.device,dtype=p.dtype,requires_grad=True)
    bias=torch.tensor(-5.,device=p.device,dtype=p.dtype,requires_grad=True)
    optimizer=torch.optim.LBFGS([scale,bias],max_iter=200,line_search_fn='strong_wolfe',tolerance_grad=1e-10,tolerance_change=1e-12)
    def closure():
        optimizer.zero_grad();logits=scale.clamp(torch.exp(torch.tensor(-4.)).item(),torch.exp(torch.tensor(6.)).item())*x+bias
        loss=(p*F.softplus(-logits)+n*F.softplus(logits)).sum();loss.backward();return loss
    optimizer.step(closure)
    s=float(scale.detach().clamp(float(torch.exp(torch.tensor(-4.))),float(torch.exp(torch.tensor(6.)))));b=float(bias.detach());pred=s*x+b>0
    result={'scale':s,'bias':b,'score_threshold':-b/s,'score_interval':[score_min,score_max],'positive_recall':float(positive[pred].sum()/positive.sum()),'negative_rejection':float(negative[~pred].sum()/negative.sum()),'balanced':balanced,'loss':float(closure().detach())}
    if score_min==-1. and score_max==1.:result['cosine_threshold']=-b/s
    return result
