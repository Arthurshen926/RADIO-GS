"""Explicit text-map/source-assisted execution, without a parent prediction chain."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--queries', required=True)
    p.add_argument('--mode', choices=['map', 'source'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--gpu', default='0')
    p.add_argument('--response', help='Explicit freshly computed native response for replay')
    p.add_argument('--evidence', help='Explicit SAM3 evidence directory for replay')
    p.add_argument('--checkpoint', default='checkpoints/sam3_modelscope/sam3.pt')
    identity_provider=p.add_mutually_exclusive_group()
    identity_provider.add_argument('--native-identity',action='store_true',help='Detector-conditioned native identity verification before scope union')
    identity_provider.add_argument('--regional-fallback', action='store_true', help='Retrieve and track a fresh native region for zero-detection texts')
    p.add_argument('--response-solver',choices=['backprojection','rendered'],default='backprojection')
    response_policy=p.add_mutually_exclusive_group()
    response_policy.add_argument('--source-response',action='store_true',help='Reconstruct semantic response with admitted source masks; keep independent selected membership')
    response_policy.add_argument('--source-calibration', action='store_true', help='Optional held-source scalar semantic calibration; does not change membership')
    a = p.parse_args()
    scene = json.loads(Path(a.scene).read_text())
    queries = json.loads(Path(a.queries).read_text())['queries']
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH='.', CUDA_VISIBLE_DEVICES=a.gpu, OMP_NUM_THREADS='4')
    native_python = '/root/miniconda3/envs/cybersim_agent/bin/python'
    sam_python = '/root/miniconda3/envs/sam3/bin/python'
    response = a.response or str(out/'map/response.pt')
    if not a.response:
        subprocess.run([native_python, 'scripts/research/query_consolidated_map.py',
                        '--scene', a.scene, '--queries', a.queries, '--output', str(out/'map')],
                       env=env, check=True)
    if a.response:
        import torch
        supplied = torch.load(response, map_location='cpu', weights_only=False)
        if supplied['texts'] != [q['text'] for q in queries]:
            raise ValueError('Explicit response does not belong to the requested text queries')
        del supplied
    if a.mode == 'map':
        return
    evidence = a.evidence or str(out/'detections')
    if not a.evidence:
        subprocess.run([sam_python, 'scripts/research/detect_consolidated_sources.py',
                        '--scene', a.scene, '--queries', a.queries, '--checkpoint', a.checkpoint,
                        '--output', evidence], env=env, check=True)
    tracks = [None for _ in queries]
    if a.regional_fallback:
        seed_out = out/'seeds'
        subprocess.run([native_python, 'scripts/research/retrieve_consolidated_seeds.py',
                        '--scene', a.scene, '--response', response, '--evidence', evidence,
                        '--output', str(seed_out)], env=env, check=True)
        seed_manifest = seed_out/'seeds.json'
        ids = json.loads(seed_manifest.read_text())['query_indices']
        if ids:
            subprocess.run([sam_python, 'scripts/research/track_full_text_sources_next79.py',
                            '--scene-input', a.scene, '--queries', a.queries,
                            '--seed-manifest', str(seed_manifest), '--checkpoint', a.checkpoint, '--key', scene['scene'],
                            '--output-base', str(out/'tracks'), '--raster-scale', '2',
                            '--query-indices', *map(str, ids)], env=env, check=True)
            for i in ids:
                tracks[i] = str(out/'tracks'/scene['scene']/f'{i}.pt')
    manifest = dict(scene=a.scene, response=response, detections=evidence,
                    tracks=tracks, base_raster=[96, 128],
                    evidence_contract='current category detections and explicitly generated regional tracks; no inherited membership')
    inputs = out/'inputs.json'
    inputs.write_text(json.dumps(manifest, indent=2)+'\n')
    if a.native_identity:
        record_out=out/'native_identity'
        subprocess.run([native_python,'scripts/research/filter_source_identity.py',
                        '--inputs',str(inputs),'--output',str(record_out),
                        '--heldout-stride','0','--record-only'],env=env,check=True)
        evidence_out=out/'identity_evidence'
        subprocess.run([native_python,'scripts/research/condition_source_identity.py',
                        '--inputs',str(inputs),'--identity-record',str(record_out/'identity.json'),
                        '--output',str(evidence_out)],env=env,check=True)
        inputs=evidence_out/'inputs.json'
    subprocess.run([native_python, 'scripts/research/decode_expanded_extent_next72.py',
                    '--inputs', str(inputs), '--key', scene['scene'], '--output-base', str(out/'sets'),
                    '--mode', 'admission', '--scope-initialization', 'source',
                    '--pixel-loss', 'balanced', '--support-prior', 'footprint',
                    '--raster-scale', '2', '--steps', '240'], env=env, check=True)
    if a.source_response:
        subprocess.run([native_python,'scripts/research/fit_consolidated_response.py','--inputs',str(inputs),
                        '--readout',str(out/'sets'/scene['scene']/'typed_readout.pt'),'--output',str(out/'reconstructed_response'),
                        '--policies','native_protected','--solver',a.response_solver],env=env,check=True)
    if a.source_calibration:
        subprocess.run([native_python, 'scripts/research/calibrate_consolidated_response.py',
                        '--scene', a.scene, '--readout', str(out/'sets'/scene['scene']/'typed_readout.pt'),
                        '--evidence', evidence, '--output', str(out/'calibrated')], env=env, check=True)


if __name__ == '__main__':
    main()
