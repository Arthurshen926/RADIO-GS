"""Inventory serialized capability names without loading tensors or target labels.

This audits package metadata and declared routes, not payload integrity, numeric
correctness, performance or a new combined benchmark method. Reading pickle
opcodes does not execute pickle. Historical per-family references stay separate.
"""
import hashlib
import json
import pickletools
from pathlib import Path
import re
import zipfile
from scripts.research.run_query_field_relations import read,write,receipt


ROOT=Path('output/query_field_20261012/common_extent_readiness_v1')


def metadata(path):
    with zipfile.ZipFile(path) as archive:
        names=[n for n in archive.namelist() if n.endswith('/data.pkl')]
        if len(names)!=1:raise ValueError('expected one PyTorch state metadata record')
        raw=archive.read(names[0])
    strings={arg for _,arg,_ in pickletools.genops(raw) if isinstance(arg,str)}
    requirements=dict(semantic_measure=['descriptors','pointer','index','mass','supported'],
        native_modes=['anchors','atoms','pointer','index','mass'],
        scope_topology=['views.0.incidence','views.0.neighborhood','views.0.semantic','views.0.retained'],
        extent_preparation=['views.0.index','views.0.weight','views.0.valid','views.0.footprint'],
        extent_factors=['views.0.pointer','views.0.index','views.0.alpha','views.0.visibility','views.0.atom'])
    present={k:all(k+'.'+suffix in strings for suffix in suffixes) for k,suffixes in requirements.items()}
    counts={k:len({int(m.group(1)) for s in strings if (m:=re.fullmatch(re.escape(k)+r'\.views\.(\d+)\.'+suffix,s))})
            for k,suffix in [('scope_topology','incidence'),('extent_preparation','weight'),('extent_factors','pointer')]}
    return dict(capabilities=present,view_counts=counts,pickle_metadata_sha256=hashlib.sha256(raw).hexdigest(),
                full_payload_hash_rechecked=False,tensors_loaded=False)


def main():
    ROOT.mkdir(parents=True,exist_ok=False);jobs=[]
    for j in read('output/query_field_20261012/signed_scope_v1/prediction_config.json')['jobs']:
        jobs.append(dict(family='lerf',scene=j['scene'],field=j['field'],route='new explicit text_extent; historical benchmarks retain separate routes'))
    p=Path('output/query_field_20261009/semantic_measure_functions_v2')
    for j in read(p/'design.json')['jobs']:
        jobs.append(dict(family='scannet',scene=j['name'],field=str(p/'fits'/j['name']/'field'),route='complete-menu semantic measure + historical shared member support'))
    p=Path('output/query_field_20261009/native_mode_functions_v1')
    for manifest in sorted(p.glob('fits/*/measure/field/manifest.json')):
        jobs.append(dict(family='uqis',scene=manifest.parents[2].name,field=str(manifest.parent),route='native text/local modes + historical conditional member readout'))
    p=Path('output/query_field_20261009/scope_conditioning_v1')
    for j in read(p/'design.json')['jobs']:
        cfg=read(j['configuration']['path'])
        fields={v['field'] for v in cfg['jobs']};assert len(fields)==1
        jobs.append(dict(family='nvos',scene=j['name'],field=fields.pop(),route='typed signed/reference scope variants; not the new extent engine',
            declared_programs=sorted({v['query_program'] for v in cfg['jobs']})))
    cache={};records=[]
    for j in jobs:
        path=Path(j['field']);manifest=read(path/'manifest.json');key=str(path.resolve())
        if key not in cache:cache[key]=metadata(path/'field.pt')
        records.append(dict(**j,manifest=receipt(path/'manifest.json'),tensor_bytes=manifest['tensor_bytes'],**cache[key]))
    families={}
    for family in ['lerf','scannet','nvos','uqis']:
        rows=[r for r in records if r['family']==family]
        families[family]=dict(requests_or_scenes=len(rows),unique_fields=len({r['field'] for r in rows}),
            capabilities={k:sum(r['capabilities'][k] for r in rows) for k in rows[0]['capabilities']})
    result=dict(complete=True,records=records,families=families,unique_fields=len(cache),
        code=receipt(__file__),metadata_only=True,new_benchmark_scores=False,combined_method_claim=False,
        task_adapters_not_implied_by_state_presence=True)
    write(ROOT/'audit.json',result);write('docs/experiments/2026-10-12-common-extent-readiness.json',result)
    print(json.dumps(families,indent=2));print('unique_fields',len(cache))


if __name__=='__main__':main()
