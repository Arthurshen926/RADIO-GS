"""Compact input-lineage checks and a figure for the CPU scoring audit."""
import argparse
import hashlib
import io
import os
from pathlib import Path
from urllib.request import urlopen

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from plyfile import PlyData

from scripts.research.audit_tmm_protocol_correctness import (
    REPO, SYSTEM, SCENE_GT_FRAMES, BoundedExperiment, POLICY, STUDY, read, load, sha)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='protocol_input_lineage_audit_v2')
    args=parser.parse_args();os.chdir(REPO)
    run=BoundedExperiment(STUDY,args.run,POLICY);run.preflight(2**20)
    rows={};inputs={str(Path(__file__)):sha(__file__)}
    jobs=read('configs/pipeline/system_v10.json')['jobs']
    for job in jobs:
        kind,name=job['benchmark'],job['name']
        if kind not in ('lerf','nvos','scannet'):continue
        scene=read(job['scene']); inputs[job['scene']]=sha(job['scene'])
        result=SYSTEM/kind/name/'result.json';inputs[str(result)]=sha(result)
        if kind=='lerf':
            for frame in SCENE_GT_FRAMES[name]:
                path=Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label')/name/(frame+'.json')
                inputs[str(path)]=sha(path)
            for obj in read(SYSTEM/kind/name/'evaluation/evaluation.json')['metrics']['hierarchical_pipeline']['selected']['objects']:
                inputs[obj['gt_path']]=sha(obj['gt_path'])
        else:
            inputs[job['evaluation']]=sha(job['evaluation']);spec=read(job['evaluation'])
            gt_path=spec['ground_truth'] if kind=='scannet' else spec['gt_mask']
            inputs[gt_path]=sha(gt_path)
        if kind=='scannet':
            path=Path('/mnt/pool/sqy/3d_understanding/scannet_og')/name/'points3d.ply'
            mesh=PlyData.read(path); props=list(mesh['vertex'].data.dtype.names)
            assert not {'label','instance','semantic'}.intersection(props)
            rows[name]=dict(unlabelled_mesh=str(path),vertex_properties=props,
                            original_mesh_semantic_labels=False)
            inputs[str(path)]=sha(path)
            continue
        report=read(scene['camera_report']); inputs[scene['camera_report']]=sha(scene['camera_report'])
        sources={r['role']:r for r in report['geometry_receipt']['inputs']}
        for role in ('moge3_scene_manifest','source_authority'):
            assert sha(sources[role]['path'])==sources[role]['sha256']
            inputs[sources[role]['path']]=sources[role]['sha256']
        authority=read(sources['source_authority']['path'])
        manifest=read(sources['moge3_scene_manifest']['path'])
        held=({int(f.split('_')[-1]) for f in SCENE_GT_FRAMES[name]} if kind=='lerf'
              else {read(job['evaluation'])['target_frame']})
        row=dict(heldout=sorted(held),
                 moge_source_intersection=sorted(held & {int(r['frame_index']) for r in manifest['records']}),
                 moge_authority_intersection=sorted(held & set(authority['frame_indices'])))
        assert not row['moge_source_intersection'] and not row['moge_authority_intersection']
        if kind=='lerf':
            root=SYSTEM/kind/name/'query/text/identity_evidence'
            frames={int(p.stem.split('_')[-1]) for p in root.glob('identity_*.pt')}
            assert frames==set(scene['source_frames'])
            row.update(identity_cache_frames=sorted(frames),identity_cache_equals_declared_sources=True,
                       identity_cache_heldout_intersection=sorted(frames & held))
        rows[name]=row
    upstream={}
    for name in ('image','evaluation','visualization'):
        url='https://raw.githubusercontent.com/naver/ludvig/4461fc515439bb498a75d71738a1e73cf7a452ed/utils/'+name+'.py'
        archived=STUDY/'protocol_input_lineage_audit_v1'/('ludvig_'+name+'_source.json')
        if archived.exists():
            reference=read(archived);source=reference['source_text']
            assert reference['url']==url and hashlib.sha256(source.encode()).hexdigest()==reference['sha256']
        else:
            source=urlopen(url,timeout=30).read().decode()
        upstream[name]=dict(url=url,sha256=hashlib.sha256(source.encode()).hexdigest())
        run.json('ludvig_'+name+'_source.json',dict(**upstream[name],source_text=source))
    for path in [Path(__file__),REPO/'scripts/research/audit_tmm_protocol_correctness.py']:
        run.json(path.stem+'_source.json',dict(path=str(path),sha256=sha(path),source_text=path.read_text()))
    audit=read(STUDY/'protocol_correctness_audit_v2/summary.json')
    fig,axes=plt.subplots(3,3,figsize=(10,7.7),layout='constrained')
    for i,name in enumerate(('flower','orchids','trex')):
        job=next(j for j in jobs if j['benchmark']=='nvos' and j['name']==name)
        spec=read(job['evaluation']);gt=np.asarray(Image.open(spec['gt_mask']))>127
        response=load(SYSTEM/'nvos'/name/'evaluation/response.pt')['response'].numpy()
        value=cv2.resize(response,(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_LINEAR)
        pred=value>=.5
        error=np.zeros((*gt.shape,3),np.uint8)
        error[gt & pred]=[200,200,200]
        error[pred & ~gt]=[231,111,81]
        error[gt & ~pred]=[42,157,143]
        # Display only is subsampled; all reported counts use original GT pixels.
        axes[i,0].imshow(value[::4,::4],vmin=0,vmax=1,cmap='viridis')
        axes[i,1].imshow(gt[::4,::4],cmap='gray',vmin=0,vmax=1)
        axes[i,2].imshow(error[::4,::4])
        r=audit['nvos']['rows'][name]['fixed']
        axes[i,0].set_ylabel(name)
        axes[i,2].set_title(f"IoU {100*r['iou']:.2f}; P {100*r['precision']:.2f}; R {100*r['recall']:.2f}",fontsize=10)
        for ax in axes[i]:ax.set_xticks([]);ax.set_yticks([])
    axes[0,0].set_title('Retained response (0 to 1)');axes[0,1].set_title('GT (>127)')
    fig.suptitle('Fixed threshold 0.5: gray = TP, orange = FP, teal = FN',fontsize=12)
    stream=io.BytesIO();fig.savefig(stream,format='png',dpi=130);plt.close(fig)
    payload=stream.getvalue();run._write('nvos_error_diagnosis.png',lambda f:f.write(payload),limit=len(payload))
    run.json('summary.json',dict(complete=True,inputs=inputs,rows=rows,upstream=upstream,
        scoring_audit_sha256=sha(STUDY/'protocol_correctness_audit_v2/summary.json'),
        image_sha256=sha(run.root/'nvos_error_diagnosis.png'),
        scope='source manifests, actual cache frame sets, unlabelled mesh properties; not historical read tracing',
        global_training_leakage_proof=False,default_changed=False))
    print('lineage checks',len(rows),'figure bytes',len(payload),flush=True)


if __name__=='__main__':main()
