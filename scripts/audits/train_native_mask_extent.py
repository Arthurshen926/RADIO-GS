"""Paired exact-renderer extent training at coarse and evaluation resolution."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file
from scripts.audits.train_rendered_extent import balanced_mask_loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True)
    a=p.parse_args();torch.set_num_threads(4)
    base=Path('/mnt/pool/sqy/results/RADIO-GS/output');root=base/'preserved_source_inventory_20260916'/a.scene
    parent=root/'grounded_semantic_transfer96';report=json.loads((parent/'report.json').read_text())
    assert sha256_file(parent/'field.pt')==report['field_sha256'];payload=torch.load(parent/'field.pt',weights_only=False)
    path=root/'caption_grounded_regions.pt';receipt=json.loads(path.with_suffix('.json').read_text());assert sha256_file(path)==receipt['sha256']
    teacher=torch.load(path,weights_only=False);info=json.loads((root/'teachers.json').read_text())
    assert teacher['teacher_manifest_sha256']==sha256_file(root/'teachers.json') and teacher['geometry_binding']==payload['geometry_binding']
    grpath=base/'geometry_only_repair_final_20260914'/a.scene/'report.json';gr=json.loads(grpath.read_text());g=gr['results']['independent_scale']
    geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=a.device);geo.assert_binding(payload['geometry_binding'])
    prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);observed=payload['state_dict']['observed'].to(a.device)
    groups=report['groups'];native={}
    for frame in info['train_frames']:
        vi=info['frames'].index(frame);ids=[j for j,g in enumerate(groups) if vi in g['views']]
        if not ids:continue
        masks=[]
        for j in ids:
            members=[i for i in groups[j]['regions'] if int(teacher['proposal_views'][i])==vi]
            layers=[]
            for i in members:
                r=teacher['records'][i];assert r['frame']==frame
                layers.append(torch.from_numpy(np.unpackbits(r['packed_native'].numpy(),axis=-1)[:,:r['native_shape'][1]].copy()).bool())
            masks.append(torch.stack(layers).any(0))
        native[frame]=(ids,torch.stack(masks))
    for width in [128,480]:
        torch.manual_seed(42)
        architecture=dict(payload['architecture'],grounded_transfer=dict(payload['architecture']['grounded_transfer'],residual_dim=128))
        model=ElementRetrievalField(**architecture).to(a.device);missing=model.load_state_dict(payload['state_dict'],strict=False)
        assert not missing.unexpected_keys and all(k.startswith('grounded_transfer.residual.') for k in missing.missing_keys)
        model.requires_grad_(False);head=model.grounded_transfer;head.decoder.requires_grad_(True);head.residual.requires_grad_(True)
        z=head.features(model.latent).detach();views=[]
        def render(colors,v):
            return rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.reshape(-1),colors=colors,
                viewmats=v['viewmat'],Ks=v['k'],width=width,height=v['height'],near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)[:2]
        for frame,(ids,mask) in native.items():
            camera=prior.cameras[frame];height=round(camera.height*width/camera.width);k=camera.intrinsic.to(a.device).clone();k[0]*=width/camera.width;k[1]*=height/camera.height
            target=F.interpolate(mask.float()[:,None],size=(height,width),mode='area')[:,0].flatten(1).T.to(a.device)
            v=dict(frame=frame,ids=torch.tensor(ids,device=a.device),target=target,height=height,k=k[None],viewmat=camera.world_to_camera.to(a.device)[None])
            with torch.no_grad():
                image,alpha=render(observed.float()[:,None],v);mass=image[0,...,0].flatten();full=alpha[0,...,0].flatten()
                v.update(mass=mass,valid=(mass/full.clamp_min(1e-8)>.5)&(full>1e-8))
            views.append(v)
        def predict(ids):
            hidden=head.residual[1](head.residual[0](z))
            raw=F.linear(z,head.decoder.weight[ids],head.decoder.bias[ids])+F.linear(hidden,head.residual[2].weight[ids],head.residual[2].bias[ids])
            return raw.clamp(1e-6,1-1e-6)*observed[:,None]
        def score():
            values=[]
            with torch.no_grad():
                for v in views:
                    image,_=render(predict(v['ids']),v);pred=image[0].flatten(0,1)/v['mass'][:,None].clamp_min(1e-8)>.5
                    target=v['target']>.5;valid=v['valid'][:,None]
                    values.extend(((pred&target&valid).sum(0)/((pred|target)&valid).sum(0).clamp_min(1)).tolist())
            return sum(values)/len(values)
        history=[dict(step=0,source_render_iou=score())];optimizer=torch.optim.Adam([p for p in head.parameters() if p.requires_grad],lr=.001)
        for step in range(400):
            v=views[step%len(views)];cols=torch.randperm(len(v['ids']),device=a.device)[:8];ids=v['ids'][cols]
            image,_=render(predict(ids),v);pred=(image[0].flatten(0,1)/v['mass'][:,None].clamp_min(1e-8)).clamp(1e-6,1-1e-6)
            loss=balanced_mask_loss(pred,v['target'][:,cols],v['valid']);optimizer.zero_grad();loss.backward();optimizer.step()
            if (step+1)%100==0:
                history.append(dict(step=step+1,source_render_iou=score(),last_loss=float(loss.detach())))
                print(a.scene,width,history[-1],flush=True)
        state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        for k,v in payload['state_dict'].items():
            if not k.startswith('grounded_transfer.decoder.'):assert torch.equal(v,state[k]),k
        out=root/f'source_native_mask_w{width}_96';out.mkdir(exist_ok=False);torch.save(dict(payload,architecture=architecture,state_dict=state),out/'field.pt')
        r=dict(report,schema='native_mask_extent_resolution.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=report['field_sha256'],
               teacher_sha256=receipt['sha256'],geometry_report_sha256=sha256_file(grpath),source_only=True,benchmark_data_used=False,
               all_parent_tensors_unchanged=False,all_parent_field_and_routing_tensors_unchanged=True,residual_dim=128,width=width,steps=400,
               seed=42,learning_rate=.001,history=history,checkpoint_selection='fixed final step',target_resampling='native binary mask union then area resize',
               implementation_sha256=sha256_file(__file__),readout_implementation_sha256=sha256_file('radio_gs/models/grounded_semantic_transfer.py'))
        (out/'report.json').write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()
