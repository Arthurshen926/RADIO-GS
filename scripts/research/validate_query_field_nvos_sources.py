"""Bind every legal NVOS native cache to query-free SAM and original RGB."""
from pathlib import Path
import json
from collections import Counter
import torch
import numpy as np
from PIL import Image
from scripts.research.build_query_field_nvos_missing_sources import ROOT
from scripts.research.run_query_field_relations import read,write,receipt,check
from radio_gs.query_field.observations import source_sam_metadata


def main():
    complete=read(ROOT/'cache_complete.json');plan=read(ROOT/'preparation.json');reports=[]
    output=ROOT/'validation/derived_source_binding_v2';output.mkdir(parents=True,exist_ok=False)
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
            binding=dict(kind='original_bytes',native_rgb=rgb)
            if source_hash!=rgb['sha256']:
                # The existing cohort builder explicitly declares this transform.
                # No resize search, tolerance or image-content similarity is used.
                native_rgb=Path(metadata['image'])
                expected=path.parent.parent/'rgb'/f'frame_{frame:05d}.png'
                if native_rgb.resolve()!=expected.resolve():raise ValueError('unknown derived RGB authority')
                derived=receipt(native_rgb)
                if derived['sha256']!=source_hash:raise ValueError('SAM/derived RGB bytes mismatch')
                image=Image.open(rgb['path']).convert('RGB');original_hw=[image.height,image.width]
                image.thumbnail((1008,1008),Image.Resampling.LANCZOS)
                with Image.open(native_rgb) as other:actual=np.array(other.convert('RGB'))
                if not np.array_equal(np.array(image),actual):raise ValueError('declared RGB thumbnail pixels differ')
                binding=dict(kind='declared_thumbnail_exact_pixels',native_rgb=derived,original_hw=original_hw,
                    native_hw=list(actual.shape[:2]),transform='PIL RGB thumbnail((1008,1008),LANCZOS)',
                    declaration=receipt(Path('scripts/research/prepare_query_field_five_benchmarks.py')))
            if path.parent.joinpath('encoding.json').exists():
                encoding=read(path.parent/'encoding.json');entry=next(e for e in encoding['frames'] if e['frame']==frame)
                check(dict(path=str(path),sha256=entry['output_sha256']))
                check(dict(path=str(sam_path),sha256=entry['sam_sha256']))
                if entry['rgb_sha256']!=binding['native_rgb']['sha256']:raise ValueError('native encoded RGB differs')
            ids=region['native_proposal_indices'];descriptors=region['descriptors'];count=len(ids)
            if count==0 or descriptors.ndim!=2 or len(descriptors)!=count or not bool(torch.isfinite(descriptors).all()):raise ValueError('invalid native vectors')
            if int(ids.min())<0 or int(ids.max())>=len(sam['packed_masks']) or len(ids.unique())!=count:raise ValueError('native proposal binding invalid')
            if region['native_mask_hw']!=list(sam['mask_shape']):raise ValueError('native raster binding differs')
            if frame in job['excluded_development']:raise ValueError('development source leaked')
            sources.append(dict(frame=frame,new=frame in new,regions=count,binding=binding,generation=generation,native=receipt(path),sam=receipt(sam_path),rgb=rgb))
            del sam,region
        report=dict(scene=row['scene'],frames=len(sources),new_frames=len(new),regions=sum(r['regions'] for r in sources),
            excluded_frames=job['excluded_development'],sources=sources,source_scene=job['source_scene'],queries_read=False,labels_read=False)
        write(output/(row['scene']+'.json'),report);reports.append(report);print('verified',row['scene'],len(sources),flush=True)
    if sum(r['frames'] for r in reports)!=243 or sum(r['new_frames'] for r in reports)!=123:raise ValueError('source inventory totals differ')
    contracts=Counter((r['new'],json.dumps(r['generation'],sort_keys=True)) for scene in reports for r in scene['sources'])
    report=dict(original_byte_binding_v1_passed=False,derived_source_binding_v2_passed=True,
        source_resolution_uniform=False,bindings=dict(Counter(r['binding']['kind'] for scene in reports for r in scene['sources'])),
        previous_failure=receipt(ROOT/'validation/original_byte_binding_v1/report.json'),generation_contracts=[dict(new=new,frames=count,contract=json.loads(c)) for (new,c),count in contracts.items()],
        generation_contract_uniform=len({c for new,c in contracts})==1,
        source_budget_only_control=False,complete=True,scenes=7,legal_frames=243,old_native_frames=120,new_native_frames=123,
        regions=sum(r['regions'] for r in reports),records=reports,cache_completion=receipt(ROOT/'cache_complete.json'),
        code=receipt(Path(__file__)),field_transfer_complete=False,benchmark_queries_evaluated=False,
        no_new_benchmark_performance_claim=True)
    write(output/'verified_sources.json',report);write('docs/experiments/2026-10-09-nvos-source-completion.json',report)


if __name__=='__main__':
    torch.set_num_threads(4);main()
