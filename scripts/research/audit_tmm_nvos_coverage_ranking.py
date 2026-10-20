"""Score AP on the exact N01 sealed images without rerunning or retaining arrays.

Private tmpfs hard links keep the original immutable NPY images alive while the
producer releases its names. No source manifest or image is modified. Labels
are opened only after the producer's entire cohort has been scored.
"""
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import torch

from radio_gs.query_field.bounded_experiment import BoundedExperiment
from radio_gs.training.retrieval_metrics import binary_ranking_metrics
from scripts.research.audit_tmm_protocol_correctness import REPO, read, binary_counts
from scripts.research.replay_tmm_response import STUDY, POLICY, sha
from scripts.research.tmm_ram_predictions import RamPredictionBank


def main():
    os.chdir(REPO);torch.set_num_threads(2)
    source=STUDY/'nvos_coverage_evidence_v2';manifest=read(source/'manifest.json')
    run=BoundedExperiment(STUDY,'nvos_coverage_ranking_v2',POLICY);run.preflight(262144)
    temp=Path(tempfile.mkdtemp(prefix='radiogs_nvos_ranking_',dir='/dev/shm'))
    parent=Path(manifest['volatile_prediction_directory']);entries={};limit=134217728
    run.json('manifest.json',dict(source_run=str(source),source_manifest_sha256=sha(source/'manifest.json'),
        volatile_directory=str(temp),maximum_volatile_bytes=limit,ground_truth_read=False,
        source_hashes={__file__:sha(__file__)},new_predictions=False))
    run.json('source_snapshot.json',{__file__:dict(sha256=sha(__file__),text=Path(__file__).read_text())})
    started=time.monotonic();bank=None
    try:
        while True:
            if (parent/'manifest.json').exists():
                available=read(parent/'manifest.json')
                for key,record in available.items():
                    if key in entries:continue
                    if sum(r['bytes'] for r in entries.values())+record['bytes']+65536>limit:
                        raise ValueError('volatile ranking bank exceeds limit')
                    os.link(parent/record['file'],temp/record['file'])
                    entries[key]=record
                (temp/'manifest.json').write_text(json.dumps(entries))
            if (source/'summary.json').exists():break
            if (source/'failure.json').exists() or time.monotonic()-started>7200:
                raise RuntimeError('N01 failed or timed out; no ranking GT read')
            time.sleep(.02)
        summary=read(source/'summary.json');seal=read(source/'predictions_complete.json')
        assert len(entries)==32 and summary['complete']
        bank=RamPredictionBank(directory=temp,limit=limit);rows={};gt_hashes={}
        for job in manifest['jobs']:
            name=job['name'];spec=read(job['evaluation']);gt=np.asarray(Image.open(spec['gt_mask']))
            gt_hashes[spec['gt_mask']]=sha(spec['gt_mask']);rows[name]={}
            truths={'retained_gt_gt127':gt>127,'ludvig_gt_maximum':(gt//gt.max()).astype(bool)}
            for arm in manifest['config']['arms']:
                key=str(source)+'::'+name+'_'+arm
                assert {k:v for k,v in entries[key].items() if k!='file'}==seal['predictions'][name+'_'+arm]
                response=bank.read(SimpleNamespace(root=source),name+'_'+arm,1).reshape(
                    spec['raster_wh'][1],spec['raster_wh'][0]).numpy()
                response=cv2.resize(response,(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_LINEAR)
                rows[name][arm]={}
                for track,truth in truths.items():
                    counts=binary_counts(response>=.5,truth)
                    assert counts==summary['rows'][name][arm][track]
                    metrics=binary_ranking_metrics(torch.from_numpy(response).flatten(),torch.from_numpy(truth).flatten())
                    rows[name][arm][track]=dict(average_precision=metrics['average_precision'],auroc=metrics['auroc'])
        macro={track:{arm:{metric:float(np.mean([r[arm][track][metric] for r in rows.values()]))
                          for metric in ('average_precision','auroc')} for arm in manifest['config']['arms']}
               for track in truths}
        run.json('summary.json',dict(complete=True,rows=rows,macro=macro,gt_hashes=gt_hashes,
            source_summary_sha256=sha(source/'summary.json'),prediction_seal_sha256=sha(source/'predictions_complete.json'),
            exact_prediction_digest_matches=32,all_confusion_counts_identical=True,
            default_changed=False,new_predictions=False,durable_dense_prediction_bytes=0))
        print(json.dumps(macro,indent=2),flush=True)
    except Exception as error:
        run.json('failure.json',dict(type=type(error).__name__,message=str(error),default_changed=False))
        raise
    finally:
        if bank is not None:bank.close()
        else:
            for record in entries.values():(temp/record['file']).unlink(missing_ok=True)
            (temp/'manifest.json').unlink(missing_ok=True);temp.rmdir()


if __name__=='__main__':main()
