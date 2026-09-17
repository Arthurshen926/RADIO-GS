"""Compile verified query inputs for ScanNet OVS and NVOS, without GT targets."""
import argparse,json,sys,os
from pathlib import Path
import torch
from PIL import Image
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.scannet_constants import NYU40_ID_TO_NAME,OPENGAUSSIAN_NYU40_CLASS_SPLITS
from radio_gs.utils.immutable_artifacts import sha256_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--sam-root',type=Path,required=True);p.add_argument('--identity',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:5');a=p.parse_args();torch.set_num_threads(4)
 if a.output.exists() or a.output.with_suffix('.json').exists():raise FileExistsError(a.output)
 report=json.loads((a.sam_root/'report.json').read_text());assert report['target_rgb_or_gt_opened'] is False
 manifest=Path('/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json');d=json.loads(manifest.read_text());assert sha256_file(manifest)==report['manifest_sha256'];forbidden=set()
 for s in d['scenes']:
  for r in s['frames']:
   if r['frame_id'] in s['evaluation_frame_ids']:
    for k in ['rgb_path','ground_truth','gt_mask_path']:
     if r.get(k):forbidden.add(str(Path(r[k]).resolve()))
 def deny(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)) and str(Path(os.fsdecode(args[0])).resolve()) in forbidden:raise RuntimeError('target access forbidden')
 sys.addaudithook(deny);identity=json.loads(a.identity.read_text());encoder=NativeThreeQueryEncoder(identity,a.device)
 ids=OPENGAUSSIAN_NYU40_CLASS_SPLITS['19'];names=[NYU40_ID_TO_NAME[i] for i in ids];text=encoder.encode_text(names).cpu();images=[];scenes=[]
 for s in report['scenes']:
  if not s['query_available']:continue
  p=Path(s['image_query_crop']['path']);assert sha256_file(p)==s['image_query_crop']['sha256'];images.append(encoder.encode_image(Image.open(p)).cpu());scenes.append({'scene':s['scene'],'reference_frame':s['reference_frame'],'crop_sha256':sha256_file(p),'full_signed_scribbles_must_remain_available':True,'signed_agreement':s['selection']['selected']['signed_agreement']})
 data={'schema':'joint_native_query_inputs.v1','encoder_identity':identity,'scannet':{'raw_class_ids':ids,'class_names':names,'native_text_queries':text,'splits':OPENGAUSSIAN_NYU40_CLASS_SPLITS,'gt_used_for_query_selection':False},'nvos':{'scenes':scenes,'native_image_queries':torch.cat(images),'sam_query_report_sha256':sha256_file(a.sam_root/'report.json'),'target_rgb_or_gt_opened':False,'is_complete_signed_prompt_execution':False},'new_field_metrics':False};torch.save(data,a.output);summary={'schema':data['schema'],'sha256':sha256_file(a.output),'scannet_query_count':len(ids),'nvos_query_count':len(scenes),'feature_dimension':text.shape[1],'native_query_norms':{'text_min':float(text.norm(dim=-1).min()),'text_max':float(text.norm(dim=-1).max()),'image_min':float(data['nvos']['native_image_queries'].norm(dim=-1).min())},'task_status':'query interfaces verified; new geometry/field bundles and target evaluation pending','target_inputs_used':False};a.output.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
if __name__=='__main__':main()
