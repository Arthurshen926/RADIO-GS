"""Versioned storage overlay for SHA-bound protected research parents.

The original design/config/results are immutable inputs. This is an explicit
storage-runtime revision, not a claim to pass the original code-freeze check.
"""
import argparse
import json
import os
from pathlib import Path

import torch

from radio_gs.query_field.legacy_field_resolver import FieldResolver
from radio_gs.query_field.research_storage import research_writer,validate_research_package
from radio_gs.query_field.storage import sha256,field_identity,deployment_bytes

ARMS=("decoder_rollback","protected_training","functional_native")
RUNTIME_FILES=(
    "radio_gs/query_field/legacy_field_resolver.py",
    "radio_gs/query_field/research_storage.py",
    "radio_gs/query_field/storage.py","radio_gs/query_field/shared_storage.py",
    "radio_gs/query_field/field.py","radio_gs/query_field/functional_training.py",
    "radio_gs/query_field/relation_training.py",
    "scripts/research/run_query_field_protected_v2.py",
)
PROJECT=Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as f:
        json.dump(value,f,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())


def source_path(path,output_root):
    p=Path(path)
    if not p.is_absolute() and p.parts and p.parts[0]=="output":
        return Path(output_root).joinpath(*p.parts[1:])
    return p if p.is_absolute() else PROJECT/p


def check_record(record,output_root):
    path=source_path(record['path'],output_root)
    if not path.is_file() or sha256(path)!=record['sha256']:
        raise ValueError('frozen non-code input missing or changed: '+record['path'])


def verify_source_design(path,expected_sha,resolver):
    if sha256(path)!=expected_sha:raise ValueError("original design SHA differs")
    design=read(path)
    for r in [design[k] for k in ("protocol","previous_design","previous_summary")]+design["configs"]+list(design["arms"].values()):
        check_record(r,resolver.output_root)
    return design


def strict_path(path):
    raw=Path(path)
    if not raw.is_absolute():raw=Path.cwd()/raw
    if any(p.is_symlink() for p in [raw,*raw.parents]):
        raise ValueError("symlink path components forbidden")
    return Path(os.path.abspath(raw))


def verify_original_construction(source_root,design,resolver,expected_sha=None):
    path=Path(source_root)/"construction_complete.json"
    if expected_sha is not None and sha256(path)!=expected_sha:
        raise ValueError("frozen construction ledger changed")
    ledger=read(path)
    by_key={r["key"]:r for r in ledger["records"]}
    if len(by_key)!=len(ledger['records']) or set(ledger['arms'])!=set(ARMS):raise ValueError('construction ledger duplicates/arms differ')
    if set(by_key)!={j["key"] for j in design["fields"]}:raise ValueError("construction cohort differs")
    for job in design["fields"]:
        record=by_key[job["key"]]
        expected_paths={str(Path(source_root)/"fits"/arm/job["key"]/name) for arm in ARMS for name in ["training.json","field/manifest.json"]}
        actual_paths={str(source_path(r["path"],resolver.output_root)) for r in record["reports"]+record["exports"]}
        if actual_paths!=expected_paths:raise ValueError("construction receipt paths differ")
        for r in record["reports"]+record["exports"]:check_record(r,resolver.output_root)
        check_record(job["cross_view_training"],resolver.output_root)
        previous=read(source_path(job["cross_view_training"]["path"],resolver.output_root))
        fitted=read(Path(source_root)/"fits/protected_training"/job["key"]/"training.json")
        native=read(Path(source_root)/"fits/functional_native"/job["key"]/"training.json")
        if fitted["trace"]!=previous["trace"] or not fitted["protected_buffers_unchanged"] or not fitted["protected_member_parameters_unchanged"]:
            raise ValueError("protected matched trace/state changed")
        if not native["protected_state_unchanged"]:raise ValueError("functional protected state changed")
    return sha256(path)


def prepare(source_root,source_design_sha,mapping,mapping_sha,run_root,store_root):
    research_writer(store_root=store_root)
    source_root=strict_path(source_root);run_root=strict_path(run_root)
    store_root=strict_path(store_root)
    if run_root.exists():raise FileExistsError(run_root)
    resolver=FieldResolver(mapping,expected_mapping_sha256=mapping_sha)
    design=verify_source_design(source_root/"design.json",source_design_sha,resolver)
    construction_sha=verify_original_construction(source_root,design,resolver)
    runtime_paths={source_path(r['path'],resolver.output_root) for r in design['code']}
    runtime_paths.update(PROJECT/p for p in RUNTIME_FILES)
    runtime_paths.update((PROJECT/'radio_gs/query_field').glob('*.py'))
    for candidate in [run_root,store_root]:
        if os.path.commonpath([str(candidate),str(source_root)])==str(source_root):
            raise ValueError('new run/store root must not be inside frozen source root')
    value=dict(schema="radio_gs.protected_storage_overlay.v1",source_root=str(source_root.absolute()),
        source_design_sha256=source_design_sha,mapping=str(Path(mapping).absolute()),mapping_sha256=mapping_sha,
        run_root=str(run_root.absolute()),store_root=str(Path(store_root).absolute()),
        source_construction_sha256=construction_sha,
        original_code_ledger=design["code"],original_code_checks_reexecuted=False,
        override_scope="explicit storage runtime revision; original scientific design parameters retained",
        runtime_code=[dict(path=str(p),sha256=sha256(p)) for p in sorted(runtime_paths)])
    write_new(run_root/"storage_overlay.json",value)
    return value


def frozen_design(run_root,*,store_root):
    run_root=strict_path(run_root);store_root=strict_path(store_root)
    overlay=read(run_root/'storage_overlay.json')
    source_root=strict_path(overlay['source_root'])
    for candidate in [run_root,store_root]:
        if os.path.commonpath([str(candidate),str(source_root)])==str(source_root):
            raise ValueError('new run/store root must not be inside frozen source root')
    if overlay.get("schema")!="radio_gs.protected_storage_overlay.v1":raise ValueError("overlay schema differs")
    if overlay["run_root"]!=os.path.abspath(run_root) or overlay["store_root"]!=os.path.abspath(store_root):
        raise ValueError("overlay run/store root differs")
    for r in overlay["runtime_code"]:
        if sha256(r["path"])!=r["sha256"]:raise ValueError("versioned runtime code changed")
    resolver=FieldResolver(overlay["mapping"],expected_mapping_sha256=overlay["mapping_sha256"])
    design=verify_source_design(Path(overlay["source_root"])/"design.json",overlay["source_design_sha256"],resolver)
    verify_original_construction(overlay['source_root'],design,resolver,overlay['source_construction_sha256'])
    return design,resolver,overlay


def inspect_completed(directory,resolver,*,load=True):
    directory=Path(directory);training=directory/"training.json"
    if not training.exists():
        if (directory/"field").exists():raise ValueError("partial field without completed training receipt")
        raise ValueError("missing frozen arm; no implicit retraining during resume")
    report=read(training);expected=report["export_sha256"]
    info=resolver.inspect(directory/"field",expected)
    if load:resolver.load(directory/"field",expected)
    return dict(training_path=str(training),training_sha256=sha256(training),
                old_archive_sha256=expected,resolution={k:v for k,v in info.items() if k!="source_manifest"})


def build(run_root,*,store_root,keys=None):
    """Verify/resume all existing arms; never launch training implicitly."""
    design,resolver,overlay=frozen_design(run_root,store_root=store_root)
    selected=design['fields']
    if keys is not None:
        if not keys or len(set(keys))!=len(keys) or set(keys)-{j['key'] for j in selected}:raise ValueError('invalid explicit resume keys')
        selected=[j for j in selected if j['key'] in set(keys)]
    records=[]
    for job in selected:
        for arm in ARMS:
            directory=Path(overlay["source_root"])/"fits"/arm/job["key"]
            records.append(dict(key=job["key"],arm=arm,**inspect_completed(directory,resolver)))
    result=dict(schema="radio_gs.protected_resolved_resume.v1",mapping_sha256=resolver.digest,
                source_design_sha256=overlay["source_design_sha256"],records=records,
                selected_keys=[j['key'] for j in selected],original_results_unchanged=True,training_launched=False)
    destination=Path(run_root)/"resolved_construction.json"
    if destination.exists():
        if read(destination)!=result:raise ValueError("cached resolved construction differs")
    else:write_new(destination,result)
    return result


def parent_binding(design,index,arm,resolver,overlay):
    job=design["fields"][index]
    if arm=="functional_native":
        directory=Path(overlay["source_root"])/"fits/protected_training"/job["key"]
        completed=inspect_completed(directory,resolver,load=False)
        return directory/"field",completed["old_archive_sha256"]
    record=job["parent_field"]
    check_record(job["parent_manifest"],resolver.output_root)
    return resolver.payload_path(record["path"]).parent,record["sha256"]


def fit(run_root,index,arm,*,store_root):
    design,resolver,overlay=frozen_design(run_root,store_root=store_root)
    if arm not in ARMS:raise ValueError("unknown arm")
    job=design["fields"][index]
    parent,expected=parent_binding(design,index,arm,resolver,overlay)
    # Validate the complete parent before creating new output or training.
    _,parent_manifest=resolver.load(parent,expected)
    observation_digest=sha256(source_path(job['observations'],resolver.output_root)/'observations.json')
    if parent_manifest['provenance'].get('observations_sha256')!=observation_digest:
        raise ValueError('frozen parent observations binding differs')
    output=Path(run_root)/"fits"/arm/job["key"]
    if output.exists():raise FileExistsError("explicit new fit refuses existing/partial output")
    save=research_writer(store_root=store_root)
    observations=source_path(job["observations"],resolver.output_root)
    if arm=="functional_native":
        from radio_gs.query_field.functional_training import fit as operation
        return operation(parent,observations,output,field_loader=resolver.loader(expected),save_function=save,**design["functional"])
    if arm=="protected_training":
        from radio_gs.query_field.relation_training import fit as operation
        return operation(parent,observations,output,arm="cross_view_anchor",
                         field_loader=resolver.loader(expected),save_function=save,**design["relation"])
    with torch.no_grad():
        original,_=resolver.load(parent,expected)
        cross=job["cross_view_field"];field,manifest=resolver.load(cross["path"],cross["sha256"])
        if manifest['provenance'].get('observations_sha256')!=observation_digest:raise ValueError('cross-view source binding differs')
        names={"scale_cutoffs","log_temperature","affinity_bias"}
        shared={k:original.state_dict()[k] for k in names};changed=[]
        for name,value in field.named_parameters():
            if name in names:changed.append(name);value.copy_(shared[name])
        if set(changed)!=names:raise ValueError("shared member parameter schema changed")
        exported=save(field,output/"field",dict(parent_field_sha256=cross["sha256"],
            shared_parameters_field_sha256=expected,observations_sha256=sha256(observations/"observations.json"),
            intervention="restore original shared acceptance parameters, retain cross-view Gaussian groups",
            additional_3d_labels=False,benchmark_queries_used=False))
        report=dict(arm=arm,changed_parameters=sorted(names),export_sha256=field_identity(exported),
                    field_schema=exported["schema"],export_state_identity=exported["state_identity"],
                    parent_sha256=cross['sha256'],shared_parameters_parent_sha256=expected,
                    observations_sha256=sha256(observations/'observations.json'),
                    deployment_bytes=deployment_bytes(exported),source_access_at_query=False,construction_training_steps=0)
        write_new(output/"training.json",report);return report


def verify_new_fit(run_root,index,arm,*,store_root):
    if arm not in ARMS:raise ValueError('unknown arm')
    design,resolver,overlay=frozen_design(run_root,store_root=store_root)
    parent,expected=parent_binding(design,index,arm,resolver,overlay)
    directory=Path(run_root)/"fits"/arm/design["fields"][index]["key"]
    if not (directory/"training.json").exists():raise ValueError("partial new fit without training receipt")
    report=read(directory/"training.json")
    manifest=validate_research_package(directory/"field",store_root=store_root)
    if report.get("field_schema")!=manifest["schema"] or report.get("export_state_identity")!=manifest["state_identity"] or report["export_sha256"]!=field_identity(manifest):
        raise ValueError("new fit receipt identity/schema differs")
    job=design['fields'][index]
    expected_parent=job['cross_view_field']['sha256'] if arm=='decoder_rollback' else expected
    provenance=manifest['provenance']
    observations_sha=sha256(source_path(job['observations'],resolver.output_root)/'observations.json')
    original_parent=resolver.inspect(parent,expected)['source_manifest']
    if original_parent['provenance'].get('observations_sha256')!=observations_sha:raise ValueError('frozen parent observations binding differs')
    if arm=='decoder_rollback':
        cross=resolver.inspect(job['cross_view_field']['path'],job['cross_view_field']['sha256'])['source_manifest']
        if cross['provenance'].get('observations_sha256')!=observations_sha:raise ValueError('cross-view source binding differs')
    if report.get('parent_sha256')!=expected_parent or provenance.get('parent_field_sha256')!=expected_parent:
        raise ValueError("new fit parent archive identity differs")
    if provenance.get('observations_sha256')!=observations_sha or report.get('observations_sha256')!=observations_sha:
        raise ValueError("new fit observations identity differs")
    if arm=='decoder_rollback':
        if provenance.get('shared_parameters_field_sha256')!=expected or report.get('shared_parameters_parent_sha256')!=expected:
            raise ValueError("rollback shared parent identity differs")
        if report.get('arm')!=arm:raise ValueError("rollback arm differs")
    else:
        settings=design['functional'] if arm=='functional_native' else design['relation']
        for key,value in settings.items():
            if report.get(key)!=value:raise ValueError("new fit scientific setting differs: "+key)
    if arm=='protected_training':
        if report.get('arm')!='cross_view_anchor' or provenance.get('arm')!='cross_view_anchor':
            raise ValueError('protected training arm differs')
        if provenance.get('parameter_policy')!=design['relation']['parameter_policy']:
            raise ValueError('protected training parameter policy differs')
        if not report.get('protected_buffers_unchanged') or not report.get('protected_member_parameters_unchanged'):
            raise ValueError('protected training state check failed')
    if arm=='functional_native':
        if not report.get('protected_state_unchanged'):raise ValueError('functional protected state check failed')
        maps=directory/'construction_maps.pt'
        if not maps.is_file() or sha256(maps)!=report.get('construction_maps_sha256'):
            raise ValueError('functional construction maps missing or changed')
    from radio_gs.query_field.shared_storage import load_field_v2
    load_field_v2(directory/'field',store_root=store_root)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["prepare","build","fit","verify-fit","export-legacy"])
    p.add_argument("--root",type=Path,required=True);p.add_argument("--store-root",type=Path,required=True)
    p.add_argument("--source-root",type=Path);p.add_argument("--source-design-sha256")
    p.add_argument("--mapping",type=Path);p.add_argument("--mapping-sha256")
    p.add_argument("--index",type=int);p.add_argument("--arm",choices=ARMS)
    p.add_argument('--keys',nargs='+',help='Explicit subset of frozen scene keys for I/O-bounded build verification')
    p.add_argument("--old-field");p.add_argument("--old-field-sha256");p.add_argument("--deployment",type=Path)
    p.add_argument('--max-export-bytes',type=int,default=16*1024**3)
    a=p.parse_args(argv)
    if a.stage=="prepare":
        if any(v is None for v in [a.source_root,a.source_design_sha256,a.mapping,a.mapping_sha256]):p.error("prepare requires explicit source design and mapping identities")
        result=prepare(a.source_root,a.source_design_sha256,a.mapping,a.mapping_sha256,a.root,a.store_root)
    elif a.stage=="build":result=build(a.root,store_root=a.store_root,keys=a.keys)
    elif a.stage in {"fit","verify-fit"}:
        if a.index is None or a.arm is None:p.error("fit requires index and arm")
        result=(fit if a.stage=="fit" else verify_new_fit)(a.root,a.index,a.arm,store_root=a.store_root)
    else:
        if any(v is None for v in [a.old_field,a.old_field_sha256,a.deployment]):p.error("export requires explicit old field/SHA and fresh deployment")
        _,resolver,_=frozen_design(a.root,store_root=a.store_root)
        result=resolver.export_legacy(a.old_field,a.old_field_sha256,a.deployment,max_export_bytes=a.max_export_bytes)
    print(json.dumps(result,indent=2));return result


if __name__=="__main__":main()
