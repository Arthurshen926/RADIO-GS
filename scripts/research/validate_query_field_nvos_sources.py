"""Bind every legal NVOS native cache to query-free SAM and original RGB."""
from pathlib import Path
import json
from collections import Counter
import torch
from scripts.research.build_query_field_nvos_missing_sources import ROOT
from scripts.research.run_query_field_relations import read,write,receipt,check
from radio_gs.query_field.observations import source_sam_metadata


def main():
    complete=read(ROOT/'cache_complete.json');plan=read(ROOT/'preparation.json');reports=[]
    for row in plan['jobs']:
        check(row['job']);job=read(row['job']['path']);check(job['source_scene']);scene=read(job['source_scene']['path'])
        built=read(ROOT/row['scene']/'cache_complete.json')
        for k in ['sam','native']:check(built[k])
        legal=job['full_legal_frames'];new={f['frame'] for f in job['frames']};sources=[]
        for frame in legal:
            paths=[Path(p)/f'frame_{frame:05d}.pt' for p in built['native_directories']]
            found=[p for p in paths if p.exists()]
            if len(found)!=1:raise ValueError('missing or ambiguous native frame '+str((row['scene'],frame,found)))
            path=found[0];region=torch.load(path,map_location='cpu',weights_only=False)
            if region['frame']!=frame or not region['query_independent'] or region['encoder']['path']!=job['encoder']:raise ValueError('native header differs')
            rgb=receipt(Path(scene['source_rgb'][str(frame)]));sam_path=Path(region['sam_path']);sam=torch.load(sam_path,map_location='cpu',weights_only=False)
            source_hash,_=source_sam_metadata(sam['metadata'])
            metadata=sam['metadata'];contract=metadata.get('generation_contract',metadata)
            generation={k:contract.get(k) for k in ['schema_version','grid_size','resolution','dtype','minimum_quality',
                'minimum_area_fraction','maximum_area_fraction','minimum_stability','nms_iou','maximum_masks',
                'complex_rope_buffers_preserved','model_cast_implementation_sha256']}
            if source_hash!=rgb['sha256']:raise ValueError('source RGB/SAM mismatch')
            ids=region['native_proposal_indices'];descriptors=region['descriptors'];count=len(ids)
            if count==0 or descriptors.ndim!=2 or len(descriptors)!=count or not bool(torch.isfinite(descriptors).all()):raise ValueError('invalid native vectors')
            if int(ids.min())<0 or int(ids.max())>=len(sam['packed_masks']) or len(ids.unique())!=count:raise ValueError('native proposal binding invalid')
            if region['native_mask_hw']!=list(sam['mask_shape']):raise ValueError('native raster binding differs')
            if frame in job['excluded_development']:raise ValueError('development source leaked')
            sources.append(dict(frame=frame,new=frame in new,regions=count,generation=generation,native=receipt(path),sam=receipt(sam_path),rgb=rgb))
            del sam,region
        report=dict(scene=row['scene'],frames=len(sources),new_frames=len(new),regions=sum(r['regions'] for r in sources),
            excluded_frames=job['excluded_development'],sources=sources,source_scene=job['source_scene'],queries_read=False,labels_read=False)
        write(ROOT/row['scene']/'verified_sources.json',report);reports.append(report);print('verified',row['scene'],len(sources),flush=True)
    if sum(r['frames'] for r in reports)!=243 or sum(r['new_frames'] for r in reports)!=123:raise ValueError('source inventory totals differ')
    contracts=Counter((r['new'],json.dumps(r['generation'],sort_keys=True)) for scene in reports for r in scene['sources'])
    report=dict(generation_contracts=[dict(new=new,frames=count,contract=json.loads(c)) for (new,c),count in contracts.items()],
        generation_contract_uniform=len({c for new,c in contracts})==1,
        source_budget_only_control=False,complete=True,scenes=7,legal_frames=243,old_native_frames=120,new_native_frames=123,
        regions=sum(r['regions'] for r in reports),records=reports,cache_completion=receipt(ROOT/'cache_complete.json'),
        code=receipt(Path(__file__)),field_transfer_complete=False,benchmark_queries_evaluated=False,
        no_new_benchmark_performance_claim=True)
    write(ROOT/'verified_sources.json',report);write('docs/experiments/2026-10-09-nvos-source-completion.json',report)


if __name__=='__main__':
    torch.set_num_threads(4);main()
