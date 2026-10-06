"""Cold all-eight evaluation of paired source-only scope refits on frozen geometry."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.querying.source_accepted_category import source_accepted_category_logits
from radio_gs.querying.source_refitted_category import source_refitted_category_logits
from radio_gs.scannet_constants import NYU40_ID_TO_NAME,OPENGAUSSIAN_NYU40_CLASS_SPLITS as SPLITS
from radio_gs.scripts.eval_scannet_vala_gaussian_protocol import assign_vala_pseudo_labels,volume_weighted_split_metrics
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import _read_label_ply
from radio_gs.utils.immutable_artifacts import sha256_file


BASE=Path('output/scannet_selected_visibility_next31')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true')
    parser.add_argument('--scenes',nargs='+');parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    torch.set_num_threads(4)
    training=json.loads((BASE/'registration.json').read_text())
    regpath=BASE/'evaluation_registration.json'
    if args.register:
        frozen_inputs={}
        for row in training['scenes']:
            root=Path(row['root'])
            old=root.parent.parent/('evaluation' if 'scannet_expanded_budget_next29' in str(root) else 'evaluation_complete_next28')
            frozen_inputs[row['scene']]=dict(query_sha256=sha256_file(old/'queries.pt'),
                scope_receipt_sha256=sha256_file(root/'bounded_fallback_withheld/report.json'))
        result=dict(schema='scannet_paired_refit_evaluation.next31',source_registration_sha256=sha256_file(BASE/'registration.json'),
            scenes=[r['scene'] for r in training['scenes']],arms=['native','retained','selected','visible_selected'],
            frozen_inputs=frozen_inputs,
            category_policy='Keep original source scope/admission acceptance, then allow only ancestry-bound source-validated head refit. Calibration/native/geometry unchanged. Correction maximum1. No per-target scene branch selection.',
            all_denominators_retained=True,independent_test=False,
            implementations={p:sha256_file(p) for p in [__file__,'radio_gs/querying/source_refitted_category.py',
                'radio_gs/querying/source_accepted_category.py','radio_gs/scripts/eval_scannet_vala_gaussian_protocol.py']})
        with regpath.open('x') as handle:json.dump(result,handle,indent=2)
        print('evaluation registered',flush=True);return
    reg=json.loads(regpath.read_text())
    assert sha256_file(BASE/'registration.json')==reg['source_registration_sha256']
    for path,digest in reg['implementations'].items():assert sha256_file(path)==digest,path
    phase={'saved':False}
    def guard(event,values):
        if event=='open' and isinstance(values[0],(str,bytes)):
            path=str(values[0])
            if '.labels.ply' in path and not phase['saved']:raise RuntimeError('Labels before predictions')
            if any(x in path for x in ('teacher_siglip.pt','teacher_dino.pt','teacher_radio.pt','teacher_sam.pt','episodes.pt','caption_region_evidence.pt')):
                raise RuntimeError('Training arrays forbidden in cold query evaluation')
    sys.addaudithook(guard)
    for row in training['scenes']:
        if row['scene'] not in args.scenes:continue
        phase['saved']=False
        run(row,training,reg,args.device,phase)


def run(row,training,registration,device,phase):
    scene=row['scene'];root=Path(row['root']);output=BASE/scene/'evaluation';output.mkdir(exist_ok=False)
    geo_report=json.loads(Path(row['geometry_report']).read_text())
    assert sha256_file(row['geometry_report'])==row['geometry_report_sha256']
    gr=geo_report['results']['independent_scale']
    geo=FrozenGaussianGeometry.load(gr['geometry'],expected_sha256=gr['geometry_sha256'],device=device)
    identity=json.loads((root/'native_query_encoder.json').read_text())
    parent_path=root/training['parent_name']/'report.json'
    parent_receipt=json.loads(parent_path.read_text())
    assert sha256_file(parent_path)==row['parent_report_sha256']
    parent,payload=load_element_retrieval_field(parent_path.with_name('field.pt'),
        expected_sha256=row['parent_field_sha256'],geometry=geo,expected_query_encoder=identity,device=device)
    scope_path=root/'bounded_fallback_withheld/report.json'
    assert sha256_file(scope_path)==registration['frozen_inputs'][scene]['scope_receipt_sha256']
    scope_receipt=json.loads(scope_path.read_text())
    assert sha256_file(scope_path.with_name('field.pt'))==scope_receipt['field_sha256']
    adaptive='scannet_expanded_budget_next29' in str(root)
    old_evaluation=root.parent.parent/('evaluation' if adaptive else 'evaluation_complete_next28')
    querypath=old_evaluation/'queries.pt'
    assert sha256_file(querypath)==registration['frozen_inputs'][scene]['query_sha256']
    bank=torch.load(querypath,map_location='cpu',weights_only=False)
    assert bank['encoder_identity']==identity
    classes=sorted(set(sum([SPLITS[k] for k in ('19','15','10')],[])))
    assert bank['texts']==[NYU40_ID_TO_NAME[k] for k in classes]
    queries=bank['encoding'].to(device)
    with torch.no_grad():
        cosine=parent.native_cosine(queries)
        native=parent.readout_log_scale['text'].clamp(-4,6).exp()*(cosine-parent.native_query_thresholds['text'](queries))
        retained,decision=source_accepted_category_logits(native,lambda:parent.logits(queries,'text'),
            scope_receipt=scope_receipt,admission_receipt=parent_receipt,max_correction=1.)
        scores=dict(native=cosine,retained=retained);decisions={'retained':decision};refit_hashes={}
        for mode in training['modes']:
            folder=BASE/scene/mode;receipt=json.loads((folder/'report.json').read_text())
            assert receipt['mode']==mode
            execution=json.loads((BASE/scene/(mode+'.execution.json')).read_text())
            assert execution['status']=='complete' and sha256_file(folder/'report.json')==execution['report_sha256']
            model,candidate=load_element_retrieval_field(folder/'field.pt',expected_sha256=receipt['field_sha256'],
                geometry=geo,expected_query_encoder=identity,device=device)
            assert candidate['architecture']==payload['architecture']
            allowed=('grounded_transfer.decoder.','grounded_transfer.residual.')
            for name,value in payload['state_dict'].items():
                if not name.startswith(allowed) or receipt['selected_step']==0:
                    assert torch.equal(value,candidate['state_dict'][name]),(scene,mode,name)
            assert torch.equal(model.native_cosine(queries),cosine),'Native response drift'
            calls=[]
            def read_scope():
                calls.append(True);return model.logits(queries,'text')
            score,refit_decision=source_refitted_category_logits(native,read_scope,
                scope_receipt=scope_receipt,parent_admission=parent_receipt,refit_receipt=receipt,
                field_sha256=receipt['field_sha256'],max_correction=1.)
            assert len(calls)==int(decision['use_scope'])
            scores[mode]=score;decisions[mode]=dict(refit_decision,scope_forward_calls=len(calls))
            refit_hashes[mode]=dict(report_sha256=sha256_file(folder/'report.json'),field_sha256=receipt['field_sha256'])
            del model,candidate
        predictions={}
        for arm,value in scores.items():
            pred={}
            for split in ('19','15','10'):
                labels=SPLITS[split];columns=[classes.index(k) for k in labels]
                picked=torch.tensor(labels,device=device)[value[:,columns].argmax(1)]
                pred[split]=torch.where(parent.observed,picked,torch.zeros_like(picked)).cpu().numpy()
            predictions[arm]=pred;np.savez_compressed(output/(arm+'.npz'),**pred)
    # No metric report or label is read until all arms have been saved.
    expected_native=np.load(old_evaluation/'native.npz')
    expected_retained=np.load(old_evaluation/'source_accepted.npz' if adaptive else Path('output/source_accepted_runtime_next29')/scene/'predictions.npz')
    mismatch={name:{split:int((predictions[name][split]!=expected[split]).sum()) for split in ('19','15','10')}
        for name,expected in [('native',expected_native),('retained',expected_retained)]}
    assert all(v==0 for by_split in mismatch.values() for v in by_split.values()),mismatch
    prediction_receipt=dict(scene=scene,prediction_sha256={k:sha256_file(output/(k+'.npz')) for k in predictions},
        query_sha256=sha256_file(querypath),decisions=decisions,baseline_mismatches=mismatch,
        source_refit_receipts=refit_hashes,scope_receipt_sha256=sha256_file(scope_path),parent_receipt_sha256=sha256_file(parent_path))
    (output/'prediction_receipt.json').write_text(json.dumps(prediction_receipt,indent=2)+'\n')
    phase['saved']=True
    gtpath=Path('/mnt/pool/sqy/3d_understanding/scannet_og')/scene/(scene+'_vh_clean_2.labels.ply')
    xyz,labels=_read_label_ply(gtpath)
    pseudo,audit=assign_vala_pseudo_labels(geo.xyz.cpu().numpy(),geo.scales.cpu().numpy(),geo.quaternions.cpu().numpy(),
        xyz,labels,radius_factor=5.,candidate_k=1000,fallback_k=1,class_balance=True,chunk_size=512)
    weights=(geo.scales.prod(1)*geo.opacities.flatten()).cpu().numpy()
    metrics={arm:{split:volume_weighted_split_metrics(pseudo,pred[split],weights,SPLITS[split])
        for split in ('19','15','10')} for arm,pred in predictions.items()}
    report=dict(**prediction_receipt,metrics=metrics,contract='scannet-ovs-paper8-v1',independent_test=False,
        input_track=geo_report['input_track'],same_geometry_and_native_field=True,
        source_only_head_refit=True,old_source_scope_policy_retained=True,all_predictions_saved_before_gt=True,
        teacher_arrays_forbidden_at_query_time=True,geometry_sha256=gr['geometry_sha256'],gt_sha256=sha256_file(gtpath),
        pseudo_gt_audit=audit,evaluation_registration_sha256=sha256_file(BASE/'evaluation_registration.json'))
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(scene,{k:{s:round(metrics[k][s]['miou']*100,3) for s in ('19','15','10')} for k in metrics},flush=True)


if __name__=='__main__':main()
