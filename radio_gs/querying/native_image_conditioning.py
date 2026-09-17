"""Keep all authorized registered 2D signs when conditioning an image query."""
import torch
from torch.nn import functional as F
from radio_gs.models.signed_query_refinement import SignedEvidenceUnavailable, observed_seed_contributions


def condition_native_image_query(field,query,seed_rows,seed_weights,labels,*,steps=64,learning_rate=.05,prior_weight=.01):
    """Optimize only a transient native image query; caller binds seed geometry.

    A SAM reference crop initializes this query but does not replace the user's
    full signed evidence. Works with both native-decoder and legacy readouts.
    Unregistered reference images can use image retrieval, but cannot invoke
    this geometric signed-evidence path without an actual registration.
    """
    if query.ndim!=2 or len(query)!=1 or query.shape[1]!=field.query_adapters['image'].in_features or not torch.isfinite(query).all() or query.norm()<=1e-12:raise ValueError('finite nonzero native image query required')
    y=torch.as_tensor(labels,device=query.device,dtype=query.dtype)
    if len(seed_rows)!=len(seed_weights) or y.shape!=(len(seed_rows),) or len(y)<2 or not torch.isfinite(y).all() or not ((y==0)|(y==1)).all() or y.min()==y.max():raise ValueError('explicit signed registered evidence required')
    if steps<0 or learning_rate<=0 or prior_weight<0:raise ValueError('invalid query optimization settings')
    with torch.no_grad():
        features=[];weights=[]
        for i,(rows,weight) in enumerate(zip(seed_rows,seed_weights)):
            rows, weight = observed_seed_contributions(
                field, rows, weight, i, device=query.device, dtype=query.dtype)
            z=field.latent[rows].detach()
            if field.native_decoder is not None:z=field.native_decoder(z)
            features.append(F.normalize(z,dim=-1).detach());weights.append(weight)
        initial=F.normalize(query.detach(),dim=-1);adapter=field.query_adapters['image'].weight.detach()
    def response(q):
        direction=F.normalize(q if field.native_decoder is not None else F.linear(q,adapter),dim=-1)
        return torch.stack([(field.cosine_logits((z@direction.T)[:,0],'image',detach_calibration=True).sigmoid()*w).sum() for z,w in zip(features,weights)])
    def objective(q):return F.binary_cross_entropy(response(q).clamp(1e-7,1-1e-7),y)+prior_weight*(1-(F.normalize(q,dim=-1)*initial).sum())
    before=response(initial).detach();best=initial.clone();best_loss=float(objective(best).detach());satisfied=torch.where(y.bool(),before>.5,before<=.5);actual=0
    if not satisfied.all():
        with torch.enable_grad():
            q=initial.clone().requires_grad_(True);optimizer=torch.optim.Adam([q],lr=learning_rate)
            for i in range(steps):
                optimizer.zero_grad();loss=objective(q)
                if not torch.isfinite(loss):raise ValueError('nonfinite transient query loss')
                loss.backward();optimizer.step();value=float(objective(q).detach());actual=i+1
                if value<best_loss:best_loss=value;best=F.normalize(q.detach(),dim=-1).clone()
                current=response(best).detach()
                if torch.where(y.bool(),current>.5,current<=.5).all():break
    after=response(best).detach()
    return {'native_query':best,'seed_before':before,'seed_after':after,'constraints_satisfied':torch.where(y.bool(),after>.5,after<=.5),'steps':actual,'field_updated':False,'native_readout':field.native_decoder is not None}
