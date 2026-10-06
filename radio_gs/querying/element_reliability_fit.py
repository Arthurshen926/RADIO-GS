"""Differentiable sufficient statistics of the deployed source vote.

Each block is one view's exact adjoint on its touched elements. Frozen linear
projection is compiled once; training differentiates only source reliability.
All declared views/elements are retained. No new inference-time optimization.
"""
import torch


class CompiledVote(torch.autograd.Function):
    @staticmethod
    def forward(ctx,quality,blocks,element_count):
        ctx.blocks=blocks;ctx.save_for_backward(quality)
        result=torch.zeros(element_count,quality.shape[1],2,device=quality.device)
        for i,block in enumerate(blocks):
            ids=block['ids'].to(quality.device)
            coefficient=block['coefficient'].to(quality.device)
            # Last axes: positive/negative reliability, numerator/denominator.
            result.index_add_(0,ids,(coefficient*quality[i,None,:,:,None]).sum(2))
        return result

    @staticmethod
    def backward(ctx,gradient):
        quality,=ctx.saved_tensors;result=torch.zeros_like(quality)
        for i,block in enumerate(ctx.blocks):
            ids=block['ids'].to(quality.device)
            coefficient=block['coefficient'].to(quality.device)
            result[i]=(coefficient*gradient[ids,:,None,:]).sum((0,3))
        return result,None,None


def compiled_probability(quality,data):
    values=CompiledVote.apply(quality,data['blocks'],len(data['prior']))
    ratio=data['ratio'].to(quality.device)
    numerator=values[:,:,0]*ratio;mass=values[:,:,1]*ratio
    prior=data['prior'].to(quality.device);visibility=data['visibility'].to(quality.device)
    p=(numerator+.1*visibility*prior)/(mass+.1*visibility).clamp_min(1e-8)
    return torch.where(mass>0,p,prior).clamp(1e-6,1-1e-6)


def retrieval_loss(probability,counts,observed):
    """Equal-scene, class-balanced point CE and soft IoU on actual element scores.

    Counts preserve the evaluator's GT-vertex -> nearest-element direction and
    equal point weights. Unknown geometry is void, retained in the IoU denominator.
    """
    p=probability.logit().softmax(1)*observed[:,None]
    size=counts.sum(0);present=size>0
    ce=-(counts*p.clamp_min(1e-8).log()).sum(0)/size.clamp_min(1)
    intersection=(counts*p).sum(0)
    union=(counts.sum(1,keepdim=True)*p).sum(0)+size-intersection
    return .5*ce[present].mean()+.5*(1-intersection[present]/union[present].clamp_min(1)).mean()
