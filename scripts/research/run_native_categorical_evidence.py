"""Fixed joint-category source mechanism on the matched native maps."""
import argparse,json,os
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',required=True);a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
 import torch
 from radio_gs.querying.scannet_runtime import load,declared_source_views
 from radio_gs.querying.categorical_evidence import solve_categorical
 torch.set_num_threads(4);root=Path('output/native_category_transfer')/a.scene
 data=load(root/'transferred/typed_readout.pt');scene=json.loads(Path(f'configs/consolidated/scannet/{a.scene}.scene.json').read_text())
 views=declared_source_views(data['semantic_observed'].cuda(),scene,data['texts'])
 value,stats=solve_categorical(data['native_logits'].cuda(),views)
 logits=value.clamp_min(1e-12).log().cpu();data.update(scores=logits,category_logits=logits)
 out=root/'joint_category';out.mkdir(exist_ok=True);torch.save(data,out/'typed_readout.pt');(out/'result.json').write_text(json.dumps(stats,indent=2)+'\n')
if __name__=='__main__':main()
