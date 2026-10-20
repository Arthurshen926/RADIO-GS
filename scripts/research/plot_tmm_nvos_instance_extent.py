"""Capture bounded sealed arrays; visualize all N11 tasks only after scoring."""
import io
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
from PIL import Image, ImageDraw
from radio_gs.query_field.bounded_experiment import BoundedExperiment
from scripts.research.replay_tmm_response import STUDY, POLICY, sha
from scripts.research.audit_tmm_protocol_correctness import read
from scripts.research.run_tmm_support_agreement import array_sha

ARMS = ['static_native', 'union_native', 'prior_anchor_native', 'scribble_anchor_native']


def main():
    parent = STUDY / 'nvos_instance_extent_v2'; m = read(parent / 'manifest.json'); jobs = m['jobs']
    expected = {j['name'] + '_' + a for j in jobs for a in ARMS}
    source = Path(m['volatile_prediction_directory']); links = {}; start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='radiogs_instance_plot_', dir='/dev/shm') as tmp:
        temp = Path(tmp)
        while len(links) < len(expected) or not (parent / 'predictions_complete.json').exists():
            if (source / 'manifest.json').exists():
                for key, e in read(source / 'manifest.json').items():
                    name = key.rsplit('::', 1)[1]
                    if name not in expected or name in links: continue
                    assert sum(p.stat().st_size for p in temp.iterdir()) + e['bytes'] + 4096 < 128 * 2**20
                    p = temp / e['file']; os.link(source / e['file'], p); links[name] = dict(e, path=str(p))
            if (parent / 'failure.json').exists() or time.monotonic() - start > 7200: raise RuntimeError('N11 prediction failed/timed out')
            time.sleep(.25)
        seal = read(parent / 'predictions_complete.json'); assert seal['images'] == 72 and not seal['target_gt_read']
        for j in jobs:
            for a in ARMS:
                e = links[j['name'] + '_' + a]
                assert array_sha(np.load(e['path'], allow_pickle=False)) == seal['predictions'][j['name']][a]['sha256'] == e['sha256']
        while not (parent / 'summary.json').exists():
            if (parent / 'failure.json').exists() or time.monotonic() - start > 10800: raise RuntimeError('N11 scoring failed/timed out')
            time.sleep(.5)
        summary = read(parent / 'summary.json'); assert summary['complete'] and len(summary['rows']) == 8
        # Pixel palette: white background, green TP, red FP, blue FN, black text.
        palette = [255,255,255, 35,143,87, 220,62,57, 47,104,198, 0,0,0] + [0] * (768-15)
        canvas = Image.new('P', (570, 832), 0); canvas.putpalette(palette); draw = ImageDraw.Draw(canvas)
        for i, label in enumerate(['GT', 'Static', 'Unrestricted', 'v10 anchor', 'Scribble geom']): draw.text((i*114+2, 2), label, fill=4)
        for row, job in enumerate(jobs):
            spec = read(job['evaluation']); name = job['name']; w, h = spec['raster_wh']
            gtpath = spec['gt_mask']; assert sha(gtpath) == summary['gt_hashes'][gtpath]
            g = np.asarray(Image.open(gtpath).resize((w, h), Image.Resampling.NEAREST)) > 127
            draw.text((2, 20+row*101), name, fill=4)
            masks = [g] + [np.load(links[name+'_'+a]['path'], allow_pickle=False).reshape(h, w) >= .5 for a in ARMS]
            for col, b in enumerate(masks):
                code = np.zeros(g.shape, np.uint8); code[b & g] = 1; code[b & ~g] = 2; code[~b & g] = 3
                tile = Image.fromarray(code, mode='P'); tile.putpalette(palette)
                canvas.paste(tile.resize((112, 84), Image.Resampling.NEAREST), (col*114, 33+row*101))
        buf = io.BytesIO(); canvas.save(buf, format='PNG', optimize=True); data = buf.getvalue()
        run = BoundedExperiment(STUDY, 'nvos_instance_extent_spatial_v1', POLICY)
        snapshot = dict(path=__file__, sha256=sha(__file__), text=Path(__file__).read_text())
        run.preflight(len(data) + len(json.dumps(snapshot).encode()) + 4096)
        run.json('source_snapshot.json', snapshot)
        run._write('errors.png', lambda f: f.write(data), limit=len(data))
        run.json('summary.json', dict(complete=True, diagnostic_only=True, tasks=8, arms=ARMS,
            parent_summary_sha256=sha(parent / 'summary.json'), parent_seal_sha256=sha(parent / 'predictions_complete.json'),
            figure_sha256=sha(run.root / 'errors.png'), source_snapshot_sha256=sha(run.root / 'source_snapshot.json'),
            volatile_parent_links=str(temp), maximum_volatile_bytes=128*2**20,
            display='native masks with nearest GT/display resizing; scores use original GT resolution', target_gt_after_parent_scoring=True))
        print('eight-task diagnostic written', len(data), 'PNG bytes', flush=True)


if __name__ == '__main__': main()
