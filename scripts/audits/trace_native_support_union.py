"""Read-only query trace of singleton versus greedy-union support coverage."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.models.grounded_semantic_transfer import greedy_native_support_union
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);a=p.parse_args();torch.set_num_threads(4)
    root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene;path=root/'native_support_union96/field.pt'
    r=json.loads((path.parent/'report.json').read_text());assert sha256_file(path)==r['field_sha256'];payload=torch.load(path,weights_only=False)
    field=ElementRetrievalField(**payload['architecture']).to(a.device);field.load_state_dict(payload['state_dict']);field.requires_grad_(False);head=field.grounded_transfer
    previous=json.loads(Path(f'paper/artifacts/{a.scene}_native_extent_selection_trace_20260916.json').read_text())['records'];categories=[v['category'] for v in previous]
    encoder=NativeThreeQueryEncoder(payload['query_encoder'],a.device)
    with torch.no_grad():
        query=encoder.encode_text(categories).half().float();_,accepted,_=head.route(query)
        support=(field.native_cosine(query)>field.native_query_thresholds['text'](query))&field.observed[:,None]
        pieces=[]
        for latent in field.latent.split(4096):
            features=head.features(latent);pieces.append((head.decoder(features)+head.residual(features))>.5)
        masks=torch.cat(pieces)&field.observed[:,None]
        fallback=(~accepted).nonzero().flatten();union,quality,chosen=greedy_native_support_union(masks,support[:,fallback])
        records=[]
        for j,qi in enumerate(fallback.tolist()):
            target=support[:,qi];selected=chosen[j]
            def stats(mask):
                hit=(mask&target).sum();return dict(elements=int(mask.sum()),native_support_iou=float(hit/(mask|target).sum().clamp_min(1)),native_support_precision=float(hit/mask.sum().clamp_min(1)),native_support_recall=float(hit/target.sum().clamp_min(1)))
            initial=masks[:,selected[0]] if selected else torch.zeros_like(target)
            records.append(dict(category=categories[qi],native_positive_elements=int(target.sum()),candidates=selected,single=stats(initial),union=stats(union[:,j])))
    out=Path(f'paper/artifacts/{a.scene}_native_support_union_trace_20260916.json')
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(dict(diagnostic_only=True,benchmark_query_names_used=True,benchmark_masks_used=False,field_sha256=sha256_file(path),implementation_sha256=sha256_file(__file__),records=records),indent=2)+'\n');print(out,flush=True)


if __name__=='__main__':main()
