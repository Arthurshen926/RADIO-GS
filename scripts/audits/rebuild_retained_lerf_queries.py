"""Rebuild retained LERF query scores from hash-verified field/teacher assets."""
import argparse
import json
from pathlib import Path
import torch
from radio_gs.scripts.build_lerf_sam_siglip_object_posterior_scores import build, build_parser
from radio_gs.utils.immutable_artifacts import sha256_file

parser = argparse.ArgumentParser()
parser.add_argument('--source-root', type=Path, required=True)
parser.add_argument('--output-root', type=Path, required=True)
args = parser.parse_args()
args.output_root.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
results = []
for scene in ('figurines', 'ramen', 'teatime', 'waldo_kitchen'):
    old_path = args.source_root / (scene+'.pt')
    old = torch.load(old_path, map_location='cpu')
    m = old['metadata']
    names = args.output_root / (scene+'_queries.json')
    names.write_text(json.dumps(m['query_names']))
    argv = ['--scene', scene, '--query-names', ','.join(m['query_names']), '--output', str(args.output_root/(scene+'.pt')),
            '--device', 'cuda:0', '--association-mode', 'latent_proposal_marginal', '--descriptor-gate', 'query_listwise']
    inputs = {}
    for key in ('primitive_query_cache', 'membership_cache', 'proposal_teacher', 'text_embedding_cache', 'canonical_embedding_cache'):
        if sha256_file(m[key]) != m[key+'_sha256']:
            raise ValueError('retained input hash changed: '+key)
        inputs[key] = {'path': m[key], 'sha256': m[key+'_sha256']}
        argv.extend(['--'+key.replace('_','-'),m[key]])
    build(build_parser().parse_args(argv))
    new = torch.load(args.output_root/(scene+'.pt'),map_location='cpu')
    comparisons = {k: {'equal': torch.equal(old[k],new[k]), 'maximum_absolute_error': float((old[k].float()-new[k].float()).abs().max())}
                   for k in ('query_scores','identity_query_scores','valid','xyz')}
    r = {'scene':scene,'inputs':inputs,'argv':argv,'comparisons':comparisons,
         'old_score_sha256':sha256_file(old_path),'new_score_sha256':sha256_file(args.output_root/(scene+'.pt'))}
    results.append(r)
    print(scene, comparisons, flush=True)
    (args.output_root/'verification.json').write_text(json.dumps(results,indent=2)+'\n')
