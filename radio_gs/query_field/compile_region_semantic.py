"""Bind native source ROI query functions to existing field coordinates once."""
import json
from pathlib import Path
import torch
from .storage import load_field_versioned,sha256,field_identity,deployment_bytes
from .research_storage import research_writer


@torch.no_grad()
def compile_field(parent,observations,construction,output,*,store_root=None,legacy_v1=False):
    save=research_writer(store_root=store_root,legacy_v1=legacy_v1)
    field,manifest=load_field_versioned(parent,store_root=store_root)
    if field.evidence.region_semantic is not None:raise ValueError('already compiled')
    root=Path(observations);source=json.loads((root/'observations.json').read_text())
    record=json.loads(Path(construction).read_text())
    old_manifest=json.loads((Path(construction).parent/'field/manifest.json').read_text())
    if old_manifest['provenance']['observations_sha256']!=sha256(root/'observations.json'):
        raise ValueError('region construction source changed')
    # Verify that the indexed relation coordinates really are those in this field.
    old_field,verified_old=load_field_versioned(Path(construction).parent/'field',store_root=store_root)
    if field_identity(verified_old)!=field_identity(old_manifest):raise ValueError('region parent changed')
    old=old_field.state_dict()
    for key,value in field.evidence.state_dict().items():
        if not torch.equal(value,old['evidence.'+key]):raise ValueError('relation coordinates changed: '+key)
    del old,old_field
    if source['world_units_per_metre']!=field.world_units_per_metre:raise ValueError('units differ')
    train=set(source['splits']['train_frames']);dev=set(source['splits']['development_frames'])
    records={r['frame']:r for r in source['frames']};descriptors=[];scales=[];views=[];receipts=[]
    for vi,r in enumerate(record['source_records']):
        frame=r['frame'];ids=torch.tensor(r['regions'],dtype=torch.long)
        if frame not in train or frame in dev or records[frame]['split']!='train':raise ValueError('illegal source frame')
        path=root/f'view_{frame:05d}.pt';digest=sha256(path)
        if digest!=records[frame]['compiled']['sha256']:raise ValueError('source view changed')
        raw=torch.load(path,map_location='cpu',weights_only=True)
        if raw['frame']!=frame or len(ids)!=len(torch.unique(ids)):raise ValueError('region binding differs')
        descriptors.append(raw['semantic_descriptors'][ids].clone())
        scales.append(raw['scales_m'][ids]);views.append(torch.full((len(ids),),vi,dtype=torch.long))
        receipts.append(dict(frame=frame,regions=r['regions'],sha256=digest))
        del raw
    descriptors=torch.cat(descriptors)
    if not torch.equal(torch.cat(views),field.evidence.views) or not torch.equal(torch.cat(scales),field.evidence.scales):
        raise ValueError('source coordinates do not match field')
    if not torch.isfinite(descriptors).all() or descriptors.shape[1]!=field.semantic.native_dim:raise ValueError('invalid native functions')
    # The only mutation: a registered optional buffer in the same scene package.
    field.evidence.region_semantic=descriptors
    exported=save(field,Path(output)/'field',dict(parent_field_sha256=field_identity(manifest),
        observations_sha256=sha256(root/'observations.json'),region_construction_sha256=sha256(construction),
        source_records=receipts,training_steps=0,benchmark_queries_used=False,target_labels_used=False,
        same_source_frames=True,uncompressed_native_region_functions=True))
    report=dict(regions=len(descriptors),native_dim=descriptors.shape[1],source_frames=len(receipts),
        additional_tensor_bytes=descriptors.numel()*descriptors.element_size(),
        total_tensor_bytes=exported['tensor_bytes'],parent_sha256=field_identity(manifest),
        export_sha256=field_identity(exported),field_schema=exported['schema'],deployment_bytes=deployment_bytes(exported),exact_relation_binding=True,
        source_records=receipts,training_steps=0,complete_v10_detection_transfer=False)
    if exported['tensor_bytes']-manifest['tensor_bytes']!=report['additional_tensor_bytes']:raise ValueError('unexpected added state')
    (Path(output)/'construction.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
