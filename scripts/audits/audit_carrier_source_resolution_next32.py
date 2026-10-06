"""Read-only current/historical carrier budget and identical-source-mask audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.utils.mask_resampling import resize_boolean_masks_nearest_exact

OUT=Path('output/carrier_source_resolution_next32')


def summarize(rows):
    if not rows:
        return dict(count=0)
    a=np.array([r['iou'] for r in rows])
    return dict(count=len(rows), mean_iou=float(a.mean()),median_iou=float(np.median(a)),
        p10_iou=float(np.quantile(a,.1)),below_half_fraction=float((a<.5).mean()),
        mean_recall=float(np.mean([r['recall'] for r in rows])),
        mean_precision=float(np.mean([r['precision'] for r in rows])),
        erased_count=sum(r['coarse_area']==0 for r in rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--keys',nargs='+');a=p.parse_args()
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    def guard(event,values):
        if event=='open' and isinstance(values[0],(str,bytes)) and any(t in str(values[0]) for t in ('.labels.ply','/lerf_ovs/label/')):
            raise RuntimeError('Target labels forbidden')
    sys.addaudithook(guard)
    hist=json.loads(Path('output/historical_method_comparison_next32/final_audit.json').read_text())
    cohort=json.loads(Path('output/native_residual_basis_next30_v2/registration.json').read_text())['cohort']
    for row in cohort:
        if row['benchmark'] not in ('lerf','scannet') or a.keys and row['key'] not in a.keys:continue
        run(row,hist)


def run(row,hist):
    target=OUT/(row['key']+'.json');assert not target.exists()
    root=Path(row['root']);bindings={}
    def read(path):
        path=Path(path);bindings[str(path)]=sha256_file(path);return json.loads(path.read_text())
    def load(path,digest=None):
        path=Path(path);actual=sha256_file(path)
        if digest is not None:assert actual==digest,str(path)
        bindings[str(path)]=actual;return torch.load(path,map_location='cpu',weights_only=False)
    info=read(root/'teachers.json');auth=read(root/'responsibility.json')
    assert bindings[str(root/'responsibility.json')]==info['authority_sha256']
    current=load(root/info['teacher_files']['sam']['path'],info['teacher_files']['sam']['sha256'])
    scene_root=root.parent.parent
    candidates=[scene_root/'base/raw_field',scene_root/'base_full_next28/raw_field']
    parentsha=info['component_source_provenance']['parent_teacher_manifest_sha256']
    rawroot=next(path for path in candidates if (path/'teachers.json').exists() and sha256_file(path/'teachers.json')==parentsha)
    rawinfo=read(rawroot/'teachers.json');raw=load(rawroot/'teacher_sam.pt',rawinfo['teacher_files']['sam']['sha256'])
    assert torch.equal(raw['masks'],current['masks'][:len(raw['masks'])])
    assert torch.equal(raw['view_indices'],current['view_indices'][:len(raw['masks'])])
    manifest_path=rawroot.parent/'source_sam/manifest.json';manifest=read(manifest_path)
    assert bindings[str(manifest_path)]==info['sam_source_manifest_sha256']
    bysha={r['output_sha256']:r for r in manifest['images']}
    h,w=current['masks'].shape[-2:]
    assert [h,w]==[auth['metadata']['feature_height'],auth['metadata']['feature_width']]
    rows=[];views=[];collisions=[]
    for vi,receipt in enumerate(raw['source_receipts']):
        record=bysha[receipt['sam_sha256']];payload=load(record['output'],record['output_sha256'])
        H,W=payload['mask_shape'];native=np.unpackbits(payload['packed_masks'].numpy(),axis=-1,bitorder='little')[...,:W].astype(bool)
        coarse=resize_boolean_masks_nearest_exact(torch.from_numpy(native),(h,w)).numpy()
        ids=torch.where(raw['view_indices']==vi)[0];native_ids=raw['native_proposal_indices'][ids].tolist()
        assert np.array_equal(coarse[native_ids],raw['masks'][ids].numpy())
        expected=[j for j,m in enumerate(coarse) if .002<=float(m.mean())<=.6]
        assert expected==native_ids,('Same-instance area filter differs',row['key'],receipt['frame'])
        y=((2*np.arange(H)+1)*h)//(2*H);x=((2*np.arange(W)+1)*w)//(2*W)
        frame_rows=[];bycoarse={}
        for j,(fine,small) in enumerate(zip(native,coarse)):
            up=small[y[:,None],x[None,:]];inter=int((up&fine).sum());union=int((up|fine).sum());area=int(fine.sum());uparea=int(up.sum())
            # Four-connected perimeter transformed into source feature-grid units.
            vertical=float(np.count_nonzero(fine[:,1:]!=fine[:,:-1])+fine[:,0].sum()+fine[:,-1].sum())*h/H
            horizontal=float(np.count_nonzero(fine[1:,:]!=fine[:-1,:])+fine[0].sum()+fine[-1].sum())*w/W
            area_grid=area*h*w/(H*W);thickness=2*area_grid/max(vertical+horizontal,1e-12)
            r=dict(frame=receipt['frame'],native_index=j,current_parent_index=int(ids[native_ids.index(j)]) if j in native_ids else None,
                retained=j in native_ids,split='train' if receipt['frame'] in info['train_frames'] else 'source_validation',
                native_shape=[H,W],native_area=area,expected_coarse_area=area_grid,coarse_area=int(small.sum()),
                thin_proxy_2area_over_perimeter_grid=thickness,iou=inter/max(union,1),
                recall=inter/max(area,1),precision=inter/max(uparea,1))
            rows.append(r);frame_rows.append(r)
            if r['retained']:
                k=hashlib.sha256(np.packbits(small).tobytes()).hexdigest();bycoarse.setdefault(k,[]).append(j)
        for members in bycoarse.values():
            if len(members)>1 and any(not np.array_equal(native[members[0]],native[j]) for j in members[1:]):
                collisions.append(dict(frame=receipt['frame'],native_indices=members,coarse_area=int(coarse[members[0]].sum())))
        views.append(dict(frame=receipt['frame'],native_shape=[H,W],native_masks=len(native),retained=len(native_ids),
            all_native=summarize(frame_rows),retained_native=summarize([r for r in frame_rows if r['retained']])))
        print(row['key'],'frame',receipt['frame'],'native',len(native),'retained',len(native_ids),flush=True)
        del native,coarse,payload
    prior=Path('output/native_residual_basis_next30_v2')/row['key']/'map'
    mr=read(prior/'report.json');data=load(prior/'map.pt',mr['map_sha256'])
    assert data['geometry_binding']==auth['metadata']['frozen_geometry_binding']
    observed=int(data['observed'].sum());elements=len(data['observed']);del data
    historic={}
    if row['benchmark']=='lerf':
        old_source=read(Path('paper/artifacts')/f"lerf_{row['scene']}_source32_exact_mpr_rgb_authority_20260817.json")
        ref=old_source['construction']['exact_mpr_authority'];old=read(ref['path']);assert bindings[ref['path']]==ref['sha256']
        historic=dict(elements=old['num_gaussians'],source_grid=[old['metadata']['feature_height'],old['metadata']['feature_width']],
            exact_authority_views=len(old['frame_indices']),proposal_source_views=old_source['construction']['selection']['selected_count'],
            geometry='historical RGB-3DGS; same geometry not claimed',observed_count=None,authority_path=ref['path'])
    else:
        tag=row['scene'].replace('_00','')
        oldpath=next(p for p in hist['source_hashes'] if '/canonical_v2_validation/scannet_'+tag+'/raw_radio_heldout4.pt.json' in p)
        old=read(oldpath)
        def grids(d):
            found=[]
            if isinstance(d,dict):
                if 'feature_height' in d and 'feature_width' in d:found.append([d['feature_height'],d['feature_width']])
                for v in d.values():found+=grids(v)
            elif isinstance(d,list):
                for v in d:found+=grids(v)
            return found
        historic=dict(elements=old['num_gaussians'],source_grids=grids(old),observed_count=old['valid_count'],
            teacher_aggregate_views=old['num_views'],geometry='historical RGB-3DGS; equal count does not imply same support',authority_path=oldpath)
    retained=[r for r in rows if r['retained']]
    strata={name:summarize(selected) for name,selected in {
        'all_native':rows,'retained_native':retained,
        'filtered_native':[r for r in rows if not r['retained']],
        'retained_thin_proxy_le2':[r for r in retained if r['thin_proxy_2area_over_perimeter_grid']<=2],
        'retained_expected_area_lt123':[r for r in retained if r['expected_coarse_area']<123],
        'retained_expected_area_123_to1229':[r for r in retained if 123<=r['expected_coarse_area']<1229],
        'retained_expected_area_ge1229':[r for r in retained if r['expected_coarse_area']>=1229]}.items()}
    result=dict(**row,source_only=True,target_gt_used=False,gpu_used=False,training_run=False,
        current=dict(elements=elements,observed_elements=observed,observed_fraction=observed/elements,
            source_grid=[h,w],train_views=len(info['train_frames']),validation_views=len(info['validation_frames']),
            augmented_regions=len(current['masks']),native_matched_parent_regions=len(raw['masks']),
            appended_regions_without_native_comparison=len(current['masks'])-len(raw['masks']),
            geometry_source=auth['metadata']['geometry_source']),historical=historic,
        same_instance_native_index_replay_exact=True,area_filter_replay_exact=True,parent_prefix_exact=True,
        strata=strata,views=views,rows=rows,distinct_native_retained_coarse_collisions=collisions,
        inputs=bindings,implementation_sha256=sha256_file(__file__),
        caveats=['Source SAM support is not object truth.','Fixed nearest-exact down/up diagnostic is not a universal model upper bound.',
            'No geometry point-position error measured.','Observed carrier fraction is not target coverage or boundary accuracy.',
            'Thickness proxy describes mask geometry, not semantic part identity.'])
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(row['key'],'DONE',strata['retained_native'],flush=True)


if __name__=='__main__':main()
