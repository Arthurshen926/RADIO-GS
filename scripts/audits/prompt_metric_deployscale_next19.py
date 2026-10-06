"""Isolated source-only cutoffs control. No latent or head updates."""
import argparse,json,sys,subprocess,os,shutil,time
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();scene=a.scene
while True:
 ready=Path('output/full_field_rebuild_next19')/scene
 if (scene!='teatime' or Path('output/prompt_metric_teatime_metadata_ready_next19.json').exists()) and (ready/'execution.json').exists() and json.loads((ready/'execution.json').read_text()).get('complete') and (ready/'reference_online.json').exists():break
 print('waiting for complete reconstruction',scene,flush=True);time.sleep(30)
source=Path('output/full_field_rebuild_next19')/scene/'reconstruction/field';out=Path('output/prompt_metric_deployscale_next19')/scene;out.mkdir(parents=True,exist_ok=False)
for f in source.iterdir():
 if f.is_dir():shutil.copytree(f.resolve(),out/f.name,copy_function=os.link)
 else:os.link(f.resolve(),out/f.name)
registration=dict(scene=scene,passes=12,trainable=['affinity.cutoffs'],mutable_buffers=['affinity.default_scale'],geometry_frozen=True,all_latent_frozen=True,other_heads_frozen=True,source_only=True,benchmark_parameter_selection=False,promoted=False,development_category_union=True,parent_sha256=sha256_file(source/'bounded_fallback_admission/field.pt'),implementation_sha256=sha256_file(__file__))
(out/'registration.json').write_text(json.dumps(registration,indent=2))
entry=Path('scripts/audits/train_bounded_structure_exploration.py');code=entry.read_text();code=code.replace("architecture=dict(payload['architecture'],scale_affinity={'scales':scale_grid,'default_scale':2})", "architecture=dict(payload['architecture']);scale_grid=list(architecture['scale_affinity']['scales'])")
code=code.replace("structure=nn.Parameter(parent.latent[:,64:].detach().clone())", "structure=nn.Parameter(parent.latent[:,64:].detach().clone(),requires_grad=False)")
code=code.replace('parameters=[structure,*field.affinity.parameters()]','parameters=[field.affinity.cutoffs]')
code=code.replace("  destination=root/name", "  assert architecture==payload['architecture']\n  changed=[k for k,v in state.items() if not torch.equal(v.cpu(),payload['state_dict'][k].cpu())]\n  assert set(changed)<= {'affinity.cutoffs','affinity.default_scale'},changed\n  destination=root/name")
code=code.replace("report['grounded_text_supervision_sha256']","sha256_file(root/'grounded_text_supervision.pt')")
code=code.replace("'structural_training':not diagnostic", "'structural_training':False,'cutoffs_training':not diagnostic,'promoted':False,'changed_state_keys':changed")
code=code.replace("'structural_updated_rows':int((counts>0).sum()) if not diagnostic else 0", "'structural_updated_rows':0")
code=code.replace("scales[i],inverse[:-2]", "field.affinity.scales[parent.affinity.default_scale],inverse[:-2]")
code=code.replace("after=validate();selected=max(range(len(after)),key=lambda i:after[i]);field.affinity.default_scale.fill_(selected)","after=validate();selected=int(parent.affinity.default_scale);field.affinity.default_scale.fill_(selected)")
code=code.replace("direct_state['affinity.default_scale'].fill_(max(range(len(before)),key=lambda i:before[i]))", "direct_state['affinity.default_scale'].fill_(int(parent.affinity.default_scale))")
code=code.replace("{'affinity.cutoffs','affinity.default_scale'}", "{'affinity.cutoffs'}")
registration.update(mutable_buffers=[],training_scale='fixed parent deployment scale; no target mask size as training input',validation_scale='fixed parent deployment scale, no scale selection')
(out/'registration.json').write_text(json.dumps(registration,indent=2))

gp=Path('output/depth_opacity_repair_20260918')/scene/'geometry_report.json';geo=json.loads(gp.read_text())['results']['independent_scale']['geometry']
sys.argv=[str(entry),'--root',str(out),'--geometry',geo,'--parent','bounded_fallback_admission','--name','prompt_metric_only','--variant','explicit','--passes','12']
exec(compile(code,str(entry),'exec'),{'__name__':'__main__','__file__':str(entry)})
cmd=[sys.executable,'scripts/audits/audit_current_query_routes.py','--task','reference','--root',str(out),'--scene',scene,'--geometry-report',str(gp),'--runs','prompt_metric_only','--device','cuda:0','--output',str(out/'reference_replay.json'),'--encoding-replay',str(Path('output/full_field_rebuild_next19')/scene/'reference_online.encodings.pt')]
subprocess.run(cmd,check=True)
old=json.loads((Path('output/full_field_rebuild_next19')/scene/'reference_online.json').read_text())['runs']['bounded_fallback_admission'];new=json.loads((out/'reference_replay.json').read_text())['runs']['prompt_metric_only']
for oldrow,newrow in zip(old['observations'],new['observations']):
 for modality in ['text','image']:assert oldrow[modality]==newrow[modality],modality
summary=dict(registration,before=old['macro'],after=new['macro'],text_image_observation_metrics_exact=True,training=json.loads((out/'prompt_metric_only/report.json').read_text()))
(out/'summary.json').write_text(json.dumps(summary,indent=2));print('COMPLETE',scene,old['macro'],new['macro'],flush=True)
