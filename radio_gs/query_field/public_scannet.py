"""Complete-menu category competition on the same Gaussian member field."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .adapters import NativeQueryEncoder
from .storage import load_field_versioned, sha256, field_identity, deployment_bytes
from .research_storage import method_binding
from .public_artifacts import persist


@torch.no_grad()
def predict(field_path, query_path, output, encoder_path, device='cuda', query_program='original', *, field_loader=None, store_root=None, artifact_writer=None):
    from .query_program import PROGRAM
    if query_program not in {'original', PROGRAM}: raise ValueError('unknown query program')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    queries=json.loads(Path(query_path).read_text())['queries']
    field,manifest=(load_field_versioned(field_path,device,store_root=store_root)
                    if field_loader is None else field_loader(field_path,device))
    encoder=NativeQueryEncoder(encoder_path,device)
    native,members,receipts,vectors=[],[],[],[]
    for query in queries:
        packet=encoder.text(query['text'],task='union')
        from .query_program import query as program_query
        result=program_query(field,packet) if query_program==PROGRAM else field.query(packet)
        vectors.append(packet.semantic.flatten())
        native.append(result.semantic_scores.cpu());members.append(result.membership_scores.cpu())
        receipts.append(dict(class_id=query['class_id'],text=query['text'],selected=int(result.selected.sum()),
                             candidates=result.candidate_indices.cpu().tolist(),scale_m=result.scale))
    payload=dict(xyz=field.xyz.cpu(),observed=field.observed.cpu(),queries=queries,
                 native=torch.stack(native,1),member=torch.stack(members,1))
    if query_program == PROGRAM:
        from .category_evidence import corroborated_category_scores
        if field.semantic_measure is None:
            payload['typed'] = corroborated_category_scores(payload['native'], payload['member'])
        else:
            from radio_gs.querying.observation_readout import positive_support_rank
            prior=field.semantic_measure.probabilities(torch.stack(vectors),
                (100*payload['native'].to(device)).softmax(1),field.semantic_measure_rule).cpu()
            payload['measure_prior']=prior
            payload['typed']=positive_support_rank(prior,payload['member'])
            payload['observed']=payload['observed'] | field.semantic_measure.supported.cpu()
    persist(root/'predictions.pt',payload,'tensor',artifact_writer)
    report=dict(field_sha256=field_identity(manifest),field_binding=method_binding(field_path,manifest),field_schema=manifest['schema'],
                complete_field_payload_bytes=deployment_bytes(manifest),query_sha256=sha256(query_path),
                predictions_sha256=sha256(root/'predictions.pt'),queries=receipts,
                query_program=query_program,
                primary=('bounded native/member category evidence' if query_program == PROGRAM else
                         'shared member union per category followed by complete-menu argmax'),
                native_control='plain native class-name cosine argmax',
                source_access=False,benchmark_labels_opened=False,field_spec=manifest['spec'],encoder=encoder_path)
    persist(root/'prediction.json',report,'json',artifact_writer)
    return report


def score(predictions,evaluation,output):
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    from radio_gs.scannet_constants import compute_split_metrics
    root=Path(predictions);record=json.loads((root/'prediction.json').read_text())
    if sha256(root/'predictions.pt')!=record['predictions_sha256']:raise ValueError('prediction payload changed')
    data=torch.load(root/'predictions.pt',map_location='cpu',weights_only=True)
    spec=json.loads(Path(evaluation).read_text())
    sets={}
    for arm in ['native','member'] + (['typed'] if 'typed' in data else []):
        sets[arm]={}
        for split,ids in spec['splits'].items():
            columns=[next(i for i,q in enumerate(data['queries']) if q['class_id']==c) for c in ids]
            labels=np.asarray(ids)[data[arm][:,columns].argmax(1).numpy()]
            labels[~data['observed'].numpy()]=0
            sets[arm][split]=labels
    # Fixed element predictions are complete before the target labels are read.
    vertices=PlyData.read(spec['ground_truth'])['vertex'].data
    xyz=np.stack([vertices[k] for k in ['x','y','z']],1)
    distance,indices=cKDTree(data['xyz'].numpy()).query(xyz,workers=4)
    truth=np.asarray(vertices['label'])
    metrics={arm:{split:compute_split_metrics(labels[indices],truth,spec['splits'][split])
                  for split,labels in predictions.items()} for arm,predictions in sets.items()}
    report=dict(scene=spec['scene'],metrics=metrics,points=len(xyz),
                observed_vertex_fraction=float(data['observed'].numpy()[indices].mean()),
                mapping='fixed unrestricted nearest Gaussian centre; unobserved predicts void and stays in denominator',
                distance_quantiles_m=np.quantile(distance,[.5,.9,.99,1]).tolist(),
                predictions_sha256=record['predictions_sha256'],ground_truth_sha256=sha256(spec['ground_truth']),
                official_protocol_equivalence_verified=False,independent_final_test=False)
    with Path(output).open('x') as handle:json.dump(report,handle,indent=2)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['predict','score'])
    parser.add_argument('--field');parser.add_argument('--queries');parser.add_argument('--evaluation');parser.add_argument('--encoder')
    parser.add_argument('--predictions');parser.add_argument('--output',required=True)
    parser.add_argument('--query-program',default='original')
    parser.add_argument("--store-root", help="Explicit v2 shared blob store; v1 remains readable")
    parser.add_argument("--device", default="cuda")
    args=parser.parse_args(argv);torch.set_num_threads(4)
    result=(predict(args.field,args.queries,args.output,args.encoder,device=args.device,
                    query_program=args.query_program,store_root=args.store_root) if args.stage=='predict'
            else score(args.predictions,args.evaluation,args.output))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
