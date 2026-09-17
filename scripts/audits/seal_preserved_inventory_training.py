"""Seal rebuilt-supervision full-field training and fixed-protocol comparisons."""
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 base=Path('/mnt/pool/sqy/results/RADIO-GS/output');art=Path('paper/artifacts');out=art/'preserved_inventory_full_training_20260916.json'
 if out.exists():raise FileExistsError(out)
 result={'schema':'preserved_inventory_full_training.v1','independent_final_test':False,'joint_model_promoted':False,'final_image_readout':'source-only balanced calibration, selected before benchmark evaluation','scenes':{}}
 for scene,oldsteps in [('figurines',2106),('ramen',3240)]:
  root=base/'preserved_source_inventory_20260916'/scene;name='preserved96_protected';report=json.loads((root/name/'report.json').read_text());state=torch.load(root/name/'field.pt',weights_only=False)['state_dict'];init=torch.load(root/'source_fusion_initialization.pt',weights_only=False);valid=init['valid']
  assert sha256_file(root/name/'field.pt')==report['field_sha256']
  assert torch.equal(state['latent'][valid,:64],init['latent'][valid])
  assert torch.equal(state['native_decoder.weight'],init['basis'].T) and not state['native_decoder.bias'].any()
  assert torch.equal(state['observed'],valid)
  assert report['episode_coverage']['complete'] and all(x==0 for x in report['cold_encoded_query_max_error'].values())
  for path,digest in report['implementation_sources'].items():assert sha256_file(Path(path))==digest
  data=torch.load(root/'episodes.pt',weights_only=False);ground=torch.load(root/'grounded_text_supervision.pt',weights_only=False);prediction=torch.load(root/'source_grounded_text.pt',weights_only=False)
  assert ground['episode_sha256']==sha256_file(root/'episodes.pt') and ground['teacher_manifest_sha256']==sha256_file(root/'teachers.json')
  original=torch.load(prediction['inventory_rebinding']['source_prediction_path'],weights_only=False)
  assert torch.equal(prediction['probability'],original['probability'])
  assert ground['queries']==data['queries']['queries'] and ground['geometry_binding']==data['geometry_binding']
  textpath=art/f'{scene}_preserved96_text_20260916.json';refpath=art/f'{scene}_preserved96_reference_20260916.json'
  oldtextpath=art/f'{scene}_partitioned_system_text_20260916.json';oldrefpath=art/f'{scene}_partitioned_system_reference_20260916.json'
  text=json.loads(textpath.read_text())['runs'][name];refs=json.loads(refpath.read_text())['runs'];oldname=f'partition96_protected_{oldsteps}';oldtext=json.loads(oldtextpath.read_text())['runs'][oldname];oldref=json.loads(oldrefpath.read_text())['runs'][oldname+'_imagecal_balanced']['macro']
  row={'native_semantics_exactly_preserved':True,'native_decoder_exactly_preserved':True,'new_supervision_binding_verified':True,'original_rgb_text_predictions_exactly_preserved':True,'report_path':str(root/name/'report.json'),'report_sha256':sha256_file(root/name/'report.json'),'finalized_field_path':str(root/(name+'_imagecal_balanced')/'field.pt'),'finalized_field_sha256':sha256_file(root/(name+'_imagecal_balanced')/'field.pt'),'steps':report['steps'],'coverage':report['episode_coverage'],'field_support':report['training_coverage'],'direct_text_supervision':report['grounded_text_training_counts'],'cold_max_error':report['cold_encoded_query_max_error'],'gradients':report['gradient_norm_max'],'text_before':{k:oldtext[k] for k in ['observation_miou','average_precision_macro']},'text_after':{k:text[k] for k in ['observation_miou','average_precision_macro']},'reference_before':oldref,'reference_after':refs[name+'_imagecal_balanced']['macro'],'reference_raw_after':refs[name]['macro'],'evaluation_hashes':{str(p):sha256_file(p) for p in [textpath,refpath,oldtextpath,oldrefpath]}}
  # Fail closed if image calibration touched semantics, structural features or prompt.
  calibrated=torch.load(row['finalized_field_path'],weights_only=False)['state_dict'];assert all(torch.equal(v,calibrated[k]) for k,v in state.items() if k not in ['readout_log_scale.image','readout_bias.image'])
  result['scenes'][scene]=row
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({s:{k:r[k] for k in ['steps','text_before','text_after','reference_before','reference_after']} for s,r in result['scenes'].items()},indent=2))
if __name__=='__main__':main()
