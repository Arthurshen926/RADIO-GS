"""Metadata-only presence of existing region incidence across retained packages."""
import hashlib
import pickletools
import zipfile
from pathlib import Path
from scripts.research.run_query_field_relations import read,write,receipt,check


def main():
    inventory=Path('docs/experiments/2026-10-12-common-extent-readiness.json')
    records=[];cache={}
    for row in read(inventory)['records']:
        check(row['manifest']);path=Path(row['field'])/'field.pt';key=str(path.resolve())
        if key not in cache:
            with zipfile.ZipFile(path) as archive:
                names=[n for n in archive.namelist() if n.endswith('/data.pkl')];assert len(names)==1
                raw=archive.read(names[0])
            strings={v for _,v,_ in pickletools.genops(raw) if isinstance(v,str)}
            keys=['positive_pointer','positive_index','positive_value','views']
            cache[key]=dict(positive_incidence_buffers_present=all('evidence.'+k in strings for k in keys),
                metadata_sha256=hashlib.sha256(raw).hexdigest(),full_tensor_payload_revalidated=False)
        records.append(dict(family=row['family'],scene=row['scene'],field=row['field'],manifest=row['manifest'],**cache[key]))
    families={name:dict(entries=sum(r['family']==name for r in records),
        with_positive_incidence=sum(r['family']==name and r['positive_incidence_buffers_present'] for r in records))
        for name in ['lerf','scannet','nvos','uqis']}
    report=dict(metadata_only=True,pickle_executed=False,tensors_loaded=False,target_labels_used=False,
        group_quality_or_task_compatibility_proven=False,whole_five_method_claim=False,
        records=records,families=families,code=receipt(__file__),prior_inventory=receipt(inventory))
    write('output/query_field_20261012/common_group_readiness_v1/audit.json',report)
    write('docs/experiments/2026-10-12-common-group-readiness.json',report)
    print(families)


if __name__=='__main__':main()
