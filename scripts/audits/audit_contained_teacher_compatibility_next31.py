"""Fixed source-only counterexamples to appearance similarity implying scope identity."""
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
from radio_gs.utils.immutable_artifacts import sha256_file


torch.set_num_threads(4)
def guard(event,args):
    if event == 'open' and isinstance(args[0],(str,bytes)) and any(x in str(args[0]) for x in (
        '.labels.ply','/lerf_ovs/label/','category_evaluation','ranking_evaluation')):
        raise RuntimeError('Target evidence forbidden')
sys.addaudithook(guard)
records = []
cohort = json.loads(Path('output/native_residual_basis_next30_v2/registration.json').read_text())['cohort']
for row in cohort:
    if row['benchmark'] not in ('lerf','scannet'):
        continue
    root = Path(row['root'])
    info = json.loads((root/'teachers.json').read_text())
    def load(rec):
        path=root/rec['path'];assert sha256_file(path)==rec['sha256']
        return torch.load(path,map_location='cpu',weights_only=False)
    sam = load(info['teacher_files']['sam'])
    siglip = load(info['teacher_files']['siglip'])
    query = load(info['query_file'])
    descriptors = F.normalize(siglip['descriptors'].float().cuda(),dim=-1)
    dino = F.normalize(query['region_dino'].float().cuda(),dim=-1)
    text = F.normalize(siglip['text_embeddings'].float().cuda(),dim=-1)
    labels = (descriptors @ text.T).argmax(1)
    cases = []
    counts = dict(contained_pairs=0,high_dino_similarity_pairs=0,
        high_dino_and_distinct_siglip_pairs=0,high_dino_and_distinct_fixed_text_labels=0)
    for vi,frame in enumerate(info['frames']):
        if frame not in info['train_frames']:
            continue
        ids = torch.where(sam['view_indices']==vi)[0]
        mask = sam['masks'][ids].flatten(1).float().cuda()
        area = mask.sum(1)
        intersection = mask @ mask.T
        containment = intersection/torch.minimum(area[:,None],area[None]).clamp_min(1)
        iou = intersection/(area[:,None]+area[None]-intersection).clamp_min(1)
        included = torch.triu(torch.ones_like(iou,dtype=torch.bool),diagonal=1)&(containment>=.95)&(iou<.95)
        ds = dino[ids] @ dino[ids].T
        ss = descriptors[ids] @ descriptors[ids].T
        high = included & (ds>.8)
        distinct = high & (ss<.8)
        counts['contained_pairs'] += int(included.sum())
        counts['high_dino_similarity_pairs'] += int(high.sum())
        counts['high_dino_and_distinct_siglip_pairs'] += int(distinct.sum())
        counts['high_dino_and_distinct_fixed_text_labels'] += int((high&(labels[ids,None]!=labels[ids][None])).sum())
        for left,right in torch.nonzero(distinct).tolist():
            i,j = int(ids[left]),int(ids[right])
            cases.append(dict(frame=frame,region_ids=[i,j],area=[int(area[left]),int(area[right])],
                containment=float(containment[left,right]),iou=float(iou[left,right]),
                dino_cosine=float(ds[left,right]),siglip_cosine=float(ss[left,right]),
                teacher_fixed_text_top1=[siglip['queries'][int(labels[i])],siglip['queries'][int(labels[j])]]))
    records.append(dict(key=row['key'],counts=counts,
        examples=sorted(cases,key=lambda x:x['siglip_cosine'])[:12],
        teacher_manifest_sha256=sha256_file(root/'teachers.json')))
    print(row['key'],counts,flush=True)
result = dict(schema='source_contained_teacher_compatibility.next31',source_only=True,records=records,
    fixed_rule=dict(containment_at_least=.95,iou_below=.95,dino_above=.8,siglip_below=.8),
    same_view_source_masks_not_verified_physical_identity=True,
    text_labels_are_teacher_predictions_not_truth=True,
    conclusion='High DINO appearance similarity cannot by itself certify same scope: contained masks can have distinct semantic descriptors and native text predictions.',
    implementation_sha256=sha256_file(__file__))
with Path('output/matched_source_fusion_next31/contained_compatibility.json').open('x') as handle:
    json.dump(result,handle,indent=2)
