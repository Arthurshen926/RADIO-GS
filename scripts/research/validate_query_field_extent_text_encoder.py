"""Fresh authorized strings to the exact vectors used by the full public replay."""
from pathlib import Path
import subprocess
import torch
from radio_gs.query_field.adapters import NativeQueryEncoder
from scripts.research.run_query_field_relations import read,write,receipt,check


ROOT=Path('output/query_field_20261012/text_extent_encoder_v1')
PUBLIC=Path('output/query_field_20261012/text_extent_public_full_v1')


def main():
    ROOT.mkdir(parents=True,exist_ok=False)
    cfg=read('output/query_field_20261012/signed_scope_v1/prediction_config.json')
    public=read(PUBLIC/'design.json')
    code=[receipt(p) for p in [__file__,'radio_gs/query_field/adapters.py','radio_gs/models/native_query_encoders.py']]
    write(ROOT/'design.json',dict(code=code,public_design=receipt(PUBLIC/'design.json'),
        protocol='all 68 legal query strings and four generic negatives, exact float32 output equality; no fitting or new target scores',
        encoder=cfg['shared_encoder'],implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    encoder=NativeQueryEncoder(cfg['shared_encoder'],'cuda')
    negatives=torch.stack([encoder.text(t).semantic.cpu().float() for t in cfg['generic_negative_texts']])
    rows=[]
    for j in public['jobs']:
        query_job=next(v for v in cfg['jobs'] if v['scene']==j['scene'])
        queries=read(query_job['queries'])['queries'];assert all(q['modality']=='text' for q in queries)
        texts=[q['text'] for q in queries];assert texts==j['texts'];check(j['query'])
        expected=torch.load(j['query']['path'],weights_only=True)
        actual=torch.stack([encoder.text(t).semantic.cpu().float() for t in texts])
        row=dict(scene=j['scene'],queries=len(texts),query_file=receipt(query_job['queries']),
            fixture=j['query'],queries_exact=torch.equal(actual,expected['native_queries']),
            negatives_exact=torch.equal(negatives,expected['generic_negatives']),
            query_max_absolute=float((actual-expected['native_queries']).abs().max()),
            negative_max_absolute=float((negatives-expected['generic_negatives']).abs().max()))
        rows.append(row);print(row,flush=True)
    result=dict(passed=all(r['queries_exact'] and r['negatives_exact'] for r in rows),scenes=rows,
        design=receipt(ROOT/'design.json'),target_labels_used=False,source_scene_inputs_used=False,
        checkpoint_io_allowed=True,performance_evaluated=False,
        composed_raw_text_chain_requires_full_public_parity=True)
    write(ROOT/'audit.json',result);write('docs/experiments/2026-10-12-text-extent-encoder-audit.json',result)
    assert result['passed'],'retain the failed strict encoder audit; do not weaken equality'


if __name__=='__main__':torch.set_num_threads(4);main()
