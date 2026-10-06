"""Scene-conditioned referent identity without changing native feature spaces.

This head consumes frozen visual/text features and unlabelled region geometry.
It predicts a scalar correction to native identity logits, not new map vectors
or element membership. Whole-instance scope remains an explicit downstream step.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def region_geometry(membership,visibility,frames,xyz):
    membership=membership.astype(np.float32);xyz=np.asarray(xyz,np.float64)
    area=np.asarray(membership.sum(0)).ravel();den=np.maximum(area,1)[:,None]
    centre=np.asarray(membership.T@xyz)/den
    variance=np.maximum(np.asarray(membership.T@(xyz*xyz))/den-centre*centre,0)
    lo,hi=xyz.min(0),xyz.max(0);scale=np.maximum(hi-lo,.01)
    view=np.asarray(visibility.sum(0)).ravel()[frames]
    geometry=np.column_stack([(centre-(hi+lo)*.5)/scale,np.sqrt(variance)/scale,
        np.log(np.maximum(area,1)/len(xyz)),np.log(np.maximum(area,1)/np.maximum(view,1))])
    return geometry.astype(np.float32),area>0


class ReferentGrounder(nn.Module):
    def __init__(self,context=True,width=128,text_width=1152,native_width=1536,native_scale=1.):
        super().__init__();self.context=context;self.native_scale=native_scale
        self.visual=nn.Linear(native_width,width);self.language=nn.Linear(text_width,width)
        self.geometry=nn.Sequential(nn.Linear(8,width),nn.GELU(),nn.Linear(width,width))
        self.cross=nn.MultiheadAttention(width,4,batch_first=True,dropout=.1)
        self.scene=nn.MultiheadAttention(width,4,batch_first=True,dropout=.1)
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width);self.norm3=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Dropout(.1),nn.Linear(width*2,width))
        self.output=nn.Linear(width,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,visual,geometry,tokens,valid,pooled):
        # Keep the native ranking term at its original precision while the
        # learned correction uses mixed precision.
        with torch.autocast(device_type=visual.device.type,enabled=False):
            base=pooled.float()@visual.float().T*self.native_scale
        x=(self.visual(visual)+self.geometry(geometry))[None].expand(len(tokens),-1,-1)
        language=self.language(tokens)
        language=self.cross(x,language,language,key_padding_mask=~valid,need_weights=False)[0]
        x=self.norm1(x+language)
        if self.context:
            context=self.scene(x,x,x,need_weights=False)[0]
        else:
            # Matched pointwise control: identical value/output projections,
            # but no evidence from another regional observation.
            width=x.shape[-1]
            value=F.linear(x,self.scene.in_proj_weight[2*width:],self.scene.in_proj_bias[2*width:])
            context=self.scene.out_proj(value)
        x=self.norm2(x+context);x=self.norm3(x+self.ff(x))
        return base+self.output(x).squeeze(-1)


class SceneReferentIdentity:
    def __init__(self,checkpoint,descriptors,membership,visibility,frames,xyz,device):
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved.get('model_type')=='grounded_scene_binding':
            from .grounded_scene_binding import GroundedSceneBinding
            self.model=GroundedSceneBinding(**saved['architecture']).to(device).eval()
        else:
            self.model=ReferentGrounder(**saved['architecture']).to(device).eval()
        self.model.load_state_dict(saved['state_dict'])
        geometry,supported=region_geometry(membership,visibility,frames,xyz)
        self.ids=np.flatnonzero(supported);self.size=len(supported);self.device=device
        self.visual=torch.from_numpy(descriptors[self.ids]).to(device)
        self.geometry=torch.from_numpy(geometry[self.ids]).to(device)
        self.receipt=dict(checkpoint=str(checkpoint),training_condition=saved['training_condition'],
            supervision_target=saved['target'],
            context=saved['architecture'].get('context',True),selected_epoch=saved['epoch'],
            model_type=saved.get('model_type','referent_grounder'),
            all_supported_regions_used=True,native_coordinates_changed=False,
            output='scalar identity logits; original native semantic field and scope retained',
            source_images_at_query=False)

    @torch.inference_mode()
    def rank(self,tokens,valid,pooled):
        result=np.full((self.size,len(pooled)),-np.inf,np.float32)
        for index in range(len(pooled)):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                scores=self.model(self.visual,self.geometry,
                    tokens[index:index+1].float().to(self.device),valid[index:index+1].to(self.device),
                    torch.from_numpy(pooled[index:index+1]).to(self.device))
            result[self.ids,index]=scores[0].float().cpu().numpy()
        return result,self.receipt
