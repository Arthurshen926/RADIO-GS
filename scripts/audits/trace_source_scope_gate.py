"""Read-only development query trace; never supplies gate supervision."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--run',default='hard_source_scope_gate96');a=p.parse_args();torch.set_num_threads(4)
    root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    path=root/a.run/'field.pt';payload=torch.load(path,weights_only=False)
    field=ElementRetrievalField(**payload['architecture']).to(a.device);field.load_state_dict(payload['state_dict']);field.requires_grad_(False)
    old=json.loads(Path(f'paper/artifacts/{a.scene}_native_extent_selection_trace_20260916.json').read_text())['records']
    categories=[r['category'] for r in old];encoder=NativeThreeQueryEncoder(payload['query_encoder'],a.device)
    with torch.no_grad():
        q=encoder.encode_text(categories).half().float();h=field.grounded_transfer;index,accepted,_=h.route(q)
        text=(F.normalize(q,dim=-1)*F.normalize(h.prototypes[index],dim=-1)).sum(-1)
        visual=(F.normalize(q,dim=-1)*h.routing_scope_images[index]).sum(-1)
        score=(torch.stack([text,visual],-1)@h.routing_scope_weights+h.routing_scope_bias).sigmoid()
    records=[dict(category=c,old_accepted=old[i]['source_gate'],new_accepted=bool(accepted[i]),source_concept=old[i]['source_concept'],scope_posterior=float(score[i]),text_cosine=float(text[i]),image_cosine=float(visual[i])) for i,c in enumerate(categories)]
    out=Path(f'paper/artifacts/{a.scene}_{a.run}_trace_20260916.json')
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(dict(diagnostic_only=True,benchmark_query_names_used=True,benchmark_masks_used=False,field_sha256=sha256_file(path),records=records,implementation_sha256=sha256_file(__file__)),indent=2)+'\n')
    for r in records:
        if r['old_accepted']:print(r,flush=True)


if __name__=='__main__':main()
