"""GT-only error attribution; oracle fills are never deployed or trained on."""
import argparse,json
from pathlib import Path
import numpy as np
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.scripts.eval_scannet_vala_gaussian_protocol import assign_vala_pseudo_labels,volume_weighted_split_metrics
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import _read_label_ply
from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=Path('output/scannet_frozen_field_20260918')/a.scene
    g=json.loads((r/'geometry_report.json').read_text())['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cpu')
    xyz,labels=_read_label_ply(Path('/mnt/pool/sqy/3d_understanding/scannet_og')/a.scene/(a.scene+'_vh_clean_2.labels.ply'))
    gt,_=assign_vala_pseudo_labels(geo.xyz.numpy(),geo.scales.numpy(),geo.quaternions.numpy(),xyz,labels,radius_factor=5.,candidate_k=1000,fallback_k=1,class_balance=True,chunk_size=512)
    weight=(geo.scales.prod(1)*geo.opacities.flatten()).numpy();ids=OPENGAUSSIAN_NYU40_CLASS_SPLITS['19']
    results={}
    for name,file in [('native','predictions.npz'),('scope','object_scope_control.npz')]:
        pred=np.load(r/'official_protocol_evaluation'/file)['19'];missing=pred==0
        variants={'actual':(pred,weight),'covered_only_different_denominator':(pred,weight*(~missing)),
            'oracle_fill_missing_only':(np.where(missing,gt,pred),weight),
            'oracle_correct_covered_only':(np.where(missing,0,gt),weight)}
        results[name]={k:volume_weighted_split_metrics(gt,v,w,ids) for k,(v,w) in variants.items()}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(scene=a.scene,diagnostic_only=True,not_deployable=True,not_training_data=True,geometry_sha256=g['geometry_sha256'],results=results),indent=2)+'\n')
    print(a.scene,{k:{n:round(m['miou'],4) for n,m in v.items()} for k,v in results.items()},flush=True)


if __name__=='__main__':main()
