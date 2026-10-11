"""Materialize the canonical positive-view gate before replacing scope parts.

An exported candidate N mask is not an admitted negative when its originating
scope was inactive. Later confidence/positive replacements must not resurrect
it. This encodes the retained text-reference contract, not a universal ban on
legal negative-only prompts in other typed adapters.
"""
import torch


def admitted_negative(scope):
    p,n,c=(scope[k] for k in ('positive','negative','confidence'))
    if (p.dtype!=torch.bool or n.dtype!=torch.bool or p.shape!=n.shape
            or c.numel()!=1 or not torch.isfinite(c).all() or bool(c<0) or bool(c>1)):
        raise ValueError('invalid originating scope admission')
    return n if bool(c>0 and p.any()) else torch.zeros_like(n)
