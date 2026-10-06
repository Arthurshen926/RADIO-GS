"""Declare retained ScanNet assets without treating old answers as model state."""
import json
from pathlib import Path
import torch
from radio_gs.scannet_constants import NYU40_ID_TO_NAME, OPENGAUSSIAN_NYU40_CLASS_SPLITS

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
out=Path('configs/consolidated/scannet');items=[]
for parent in sorted(Path('output/taskwise_scope_composition_next38').glob('scannet_*/field.pt')):
 key=parent.parent.name;scene=key.removeprefix('scannet_');memory=load(parent);root=Path(memory['root'])
 info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text())
 source=json.loads((root/'source_frames.json').read_text());byframe={r['frame_idx']:r for r in source['frames']}
 gr=json.loads(Path(memory['geometry_report']).read_text());g=gr['results']['independent_scale']
 encoder=json.loads((root/'native_query_encoder.json').read_text())
 views=[dict(frame=f,identity=str(Path('output/typed_evidence_next44')/key/f'identity_{f}.pt'),contributions=str(root/auth['views'][info['frames'].index(f)]['relative_path']),source_rgb=str(Path(source['image_dir'])/byframe[f]['source_file'])) for f in info['train_frames']]
 spec=dict(scene=key,runtime_provider='legacy_category_memory',input_track='provided_unlabelled_mesh',memory=str(parent),base_field=memory['base_field'],calibration=str(parent.parent/'calibration.pt'),geometry=dict(path=g['geometry'],sha256=g['geometry_sha256']),text_encoder=encoder['siglip_root'],source_scope_receipt=str(root/'bounded_fallback_withheld/report.json'),source_admission_receipt=str(Path(memory['base_field']).parent/'report.json'),source_evidence=dict(provider='SAM3 literal text with native foreground-context evidence',raster_hw=[auth['metadata']['feature_height'],auth['metadata']['feature_width']],views=views),source_frames=info['train_frames'],source_views=len(views),full_train_source_views=len(json.loads((Path('/mnt/pool/sqy/3d_understanding/scannet_og')/scene/'transforms_train.json').read_text())['frames']),map_contract=dict(latent_dim=96,native_semantic_dim=64,native_dictionary_dim=1536,native_dictionary_entries=512,teachers=['siglip','radio','dino','sam'],runtime_query_independent_deployment_state=True,source_query_supervision_used=True,source_query_dependent_calibration=True,shared_with_latest_lerf_map_recipe=False))
 classes=sorted(set(sum(OPENGAUSSIAN_NYU40_CLASS_SPLITS.values(),[])))
 query=dict(queries=[dict(id=str(c),modality='text',intent='category',text=NYU40_ID_TO_NAME[c],class_id=c) for c in classes])
 evaluation=dict(scene=scene,ground_truth=str(Path('/mnt/pool/sqy/3d_understanding/scannet_og')/scene/(scene+'_vh_clean_2.labels.ply')),splits=OPENGAUSSIAN_NYU40_CLASS_SPLITS,protocols=['point_nearest_center','gaussian_volume_weighted_pseudolabel'],official_protocol_equivalence_verified=False,independent_test=False)
 save(out/(scene+'.scene.json'),spec);save(out/(scene+'.queries.json'),query);save(out/(scene+'.evaluation.json'),evaluation)
 items.append(dict(scene=scene,scene_config=str(out/(scene+'.scene.json')),query_config=str(out/(scene+'.queries.json')),evaluation_config=str(out/(scene+'.evaluation.json'))))
save(out/'cohort.json',dict(scenes=items,missing_semantic_maps=['scene0200_00','scene0645_00'],historical_predictions_as_inputs=False))
