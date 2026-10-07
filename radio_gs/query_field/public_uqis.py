"""Four independent public inputs query one Gaussian field; no evaluator access."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch

from .adapters import NativeQueryEncoder, point_2d, point_3d
from .contracts import QueryPacket
from .storage import load_field, sha256


@torch.no_grad()
def predict(field_path,public,scene,output,encoder_path,device='cuda'):
    public=Path(public);root=Path(output)
    for folder in ['predictions','selections','receipts','gaussians']:(root/folder).mkdir(parents=True,exist_ok=True)
    report_path=root/'receipts'/f'{scene}.json'
    if report_path.exists():raise FileExistsError(report_path)
    domain=next(d for d in json.loads((public/'scenes.json').read_text())['scene_domains'] if d['scene_id']==scene)
    xyz_path=public/domain['mesh_xyz_path']
    if sha256(xyz_path)!=domain['mesh_xyz_sha256']:raise ValueError('public mesh changed')
    xyz=np.load(xyz_path)
    # Each encoder sees one authorized input. No paired target IDs or labels are loaded.
    queries=[]
    for modality in ['text','image','point_2d','point_3d']:
        queries += [q for q in json.loads((public/'queries'/f'{modality}.json').read_text())['queries'] if q['scene_id']==scene]
    encoder=NativeQueryEncoder(encoder_path,device)
    encoded={};started=time.monotonic()
    for query in queries:
        modality=query['modality']
        if modality=='text':
            packet=encoder.text(query['expression'])
        elif modality=='image':
            image_path=public/query['crop_rgb_path']
            if sha256(image_path)!=query['crop_rgb_sha256']:raise ValueError('query image changed')
            packet=encoder.image(Image.open(image_path).convert('RGB'))
        else:continue
        encoded[query['query_id']]=(packet.semantic.cpu(),None if packet.local is None else packet.local.cpu())
    encoding_seconds=time.monotonic()-started
    del encoder
    if encoded:del packet
    gc.collect();torch.cuda.empty_cache()
    field,manifest=load_field(field_path,device)
    distance,indices=cKDTree(field.xyz.cpu().numpy()).query(xyz,workers=4)
    indices=torch.from_numpy(indices).to(device)
    rows=[]
    for query in queries:
        qid=query['query_id'];modality=query['modality'];stamp=time.monotonic()
        status='ok';reason=None
        try:
            if modality in {'text','image'}:
                semantic,local=encoded[qid]
                packet=QueryPacket(modality,semantic=semantic.to(device),local=None if local is None else local.to(device))
            elif modality=='point_2d':
                width,height=query['raster_size']
                packet=point_2d(field,query['positive_pixel_uv'],query['camera_intrinsics'],
                                torch.linalg.inv(torch.tensor(query['camera_to_world'],dtype=torch.float64)).float(),
                                [height,width],world_units_per_metre=1.)
            else:
                packet=point_3d(field,query['point_world_xyz'],maximum_distance=.1)
            result=field.query(packet)
            scores=result.membership_scores[indices].cpu().numpy().astype(np.float32)
            selected=result.selected[indices].cpu().numpy()
            diagnostics=result.diagnostics
            gaussian=dict(membership=result.membership_scores.cpu(),selected=result.selected.cpu(),
                          candidates=result.candidate_indices.cpu(),candidate_weights=result.candidate_weights.cpu(),
                          native_scores=None if result.semantic_scores is None else result.semantic_scores.cpu())
        except ValueError as exc:
            # A valid public prompt can lack trained observation support; retain zero output.
            if str(exc) not in {'support requires nonempty integer indices and weights',
                                'support requires nonempty integer indices',
                                'positive support lacks trained observation support',
                                '3D query has no Gaussian within registration tolerance',
                                'field has no observed identity candidates'}:
                raise
            status='unsupported';reason=str(exc);scores=np.zeros(len(xyz),np.float32);selected=np.zeros(len(xyz),bool)
            diagnostics={}
            gaussian=dict(membership=torch.zeros(len(field.xyz)),selected=torch.zeros(len(field.xyz),dtype=torch.bool),
                          candidates=torch.empty(0,dtype=torch.long),candidate_weights=torch.empty(0),native_scores=None)
        prediction=root/'predictions'/f'{qid}.npy';selection=root/'selections'/f'{qid}.npy'
        if prediction.exists() or selection.exists():raise FileExistsError(qid)
        np.save(prediction,scores);np.save(selection,selected)
        gaussian_path=root/'gaussians'/f'{qid}.pt';torch.save(gaussian,gaussian_path)
        rows.append(dict(query_id=qid,modality=modality,status=status,reason=reason,selected=int(selected.sum()),
                         seconds=time.monotonic()-stamp,diagnostics=diagnostics,
                         prediction_sha256=sha256(prediction),selection_sha256=sha256(selection),
                         gaussian_sha256=sha256(gaussian_path)))
        print(scene,modality,qid,status,flush=True)
    report=dict(scene=scene,field_sha256=manifest['files']['field.pt']['sha256'],field_spec=manifest['spec'],
                queries=rows,query_count=len(rows),unsupported=sum(r['status']!='ok' for r in rows),
                domain_sha256=domain['mesh_xyz_sha256'],distance_quantiles_m=np.quantile(distance,[.5,.9,.99,1]).tolist(),
                observed_vertex_fraction=float(field.observed[indices].float().mean()),
                transport='fixed nearest Gaussian centre; scores and actual set transported with the same indices',
                encoding_seconds=encoding_seconds,source_access=False,evaluator_access=False,
                target_pairing_used=False,threshold=field.threshold)
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--field',required=True);parser.add_argument('--public',required=True)
    parser.add_argument('--scene',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--encoder',required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    report=predict(args.field,args.public,args.scene,args.output,args.encoder)
    print(json.dumps({k:v for k,v in report.items() if k!='queries'},indent=2))
