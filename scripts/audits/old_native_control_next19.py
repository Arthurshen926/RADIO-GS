"""Complete missing old native control using sealed field and query, no fitting."""
import argparse,json,sys,torch
from pathlib import Path
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4)
s=a.scene;out=Path('output/old_native_control_next19')/s;out.mkdir(parents=True,exist_ok=False)
old=Path('output/graph_transfer_final_20260919')/s;cache=torch.load(old/'scores.pt',map_location='cpu');assert sha256_file(old/'scores.pt')==json.loads((old/'preparation.json').read_text())['scores_sha256']
fp=Path('output/depth_opacity_pipeline_20260918')/s/'reconstruction/field/bounded_fallback_admission/field.pt';qp=Path('output/bounded_fallback_repair_20260918')/s/'text_online.encodings.pt';payload=torch.load(fp,map_location='cpu');bank=torch.load(qp,map_location='cpu')[0];q=bank['encoding'].half().float();assert bank['texts']==cache['texts']
m=ElementRetrievalField(**payload['architecture']);m.load_state_dict(payload['state_dict']);m=m.cuda().eval();m.requires_grad_(False)
with torch.no_grad():
 native=torch.sigmoid(m.readout_log_scale['text'].clamp(-4,6).exp()*(m.native_cosine(q.cuda())-m.native_query_thresholds['text'](q.cuda())))*m.observed[:,None]
 final=m.retrieve(q.cuda(),'text')['scores'].cpu();err=float((final-cache['scores']['final']).abs().max());assert err<2e-5,err
assert torch.equal(m.observed.cpu(),cache['observed']);assert payload['geometry_binding']==cache['geometry_binding']
data=dict(cache,scores={'native_compact_calibrated':native.cpu()});torch.save(data,out/'scores.pt');gp=Path('output/depth_opacity_repair_20260918')/s/'geometry_report.json'
prep=dict(scene=s,geometry_report_sha256=sha256_file(gp),scores_sha256=sha256_file(out/'scores.pt'),old_field_sha256=sha256_file(fp),old_query_sha256=sha256_file(qp),old_final_replay_max_error=err,formula='sigmoid(clamped_exp_text_scale*(native_cosine(query)-frozen_query_threshold(query)))*observed',no_fitting=True,old_native_cache_available=False,implementation_sha256=sha256_file(__file__))
(out/'preparation.json').write_text(json.dumps(prep,indent=2))
del m,native,final;torch.cuda.empty_cache()
source=Path('scripts/audits/diagnose_lerf_native_membership.py').read_text();start=source.index('            from PIL import Image\n');end=source.index("        print(a.scene,frame",start)
source=source[:start]+"            rows.append(dict(frame=frame,query=obj.query,ap=ap,membership_iou=membership_iou,positive_support_recall=float(valid.cpu().numpy()[obj.mask].mean())))\n"+source[end:]
source=source.replace("final_prediction_disagreement_pixels=sum(r['final_prediction_disagreement_pixels'] for r in rows),final_miou=float(np.mean([r['final_iou'] for r in rows])),",'')
sys.argv=['diagnose_lerf_native_membership.py','--scene',s,'--output',str(out),'--render-only'];exec(compile(source,'scripts/audits/diagnose_lerf_native_membership.py','exec'),{'__name__':'__main__','__file__':'scripts/audits/diagnose_lerf_native_membership.py'})
source=Path('scripts/audits/graph_transfer_20260919_final_selected.py').read_text().replace("Path('output/graph_transfer_final_20260919')","Path('output/old_native_control_next19')")
sys.argv=['selected','--scene',s,'--output',str(out/'selected')];exec(compile(source,'scripts/audits/graph_transfer_20260919_final_selected.py','exec'),{'__name__':'__main__','__file__':'scripts/audits/graph_transfer_20260919_final_selected.py'})
print('COMPLETE',s,flush=True)
