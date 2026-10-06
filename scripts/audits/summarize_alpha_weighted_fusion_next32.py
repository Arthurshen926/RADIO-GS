"""Summarize the registered source-only opacity-weighting control."""
import json
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file

ROOT = Path('output/alpha_weighted_source_fusion_next32')


def main():
    registration = json.loads((ROOT / 'registration.json').read_text())
    for path, digest in registration['implementations'].items():
        assert sha256_file(path) == digest, path
    rows = []
    for key in registration['pilots']:
        path = ROOT / key / 'report.json'
        r = json.loads(path.read_text())
        assert r['registration_sha256'] == sha256_file(ROOT / 'registration.json')
        assert r['map_sha256'] == sha256_file(ROOT / key / 'map.pt')
        assert r['source_only'] and not r['target_evidence_used']
        rows.append(dict(key=key, source_admitted=r['source_admitted'],
            region_mse_change_percent={m:100*(g['mse_ratio']-1) for m,g in r['gate'].items()},
            pixel_mse_change_percent={m:{s:100*(g['mse_ratio']-1) for s,g in strata.items()} for m,strata in r['pixel_gate'].items()},
            regional_teacher_top1_change_pp={m:-100*g['top1_drop'] for m,g in r['gate'].items()},
            alpha_diagnostics=r['alpha_diagnostics'],
            source_supported_regions=sum(v['supported_regions'] for v in r['observations']),
            source_unsupported_regions=sum(v['regions']-v['supported_regions'] for v in r['observations']),
            source_valid_covered_pixels=sum(v['covered_pixels'] for v in r['observations']),
            baseline_replay_max_error=r['baseline_fusion_replay_max_error'],
            maximum_element_change=r['maximum_element_change'],
            report_sha256=sha256_file(path)))
    report = dict(schema='alpha_weighted_source_fusion_summary.next32',
        registration_sha256=sha256_file(ROOT / 'registration.json'), rows=rows,
        full_cohort_authorized=all(r['source_admitted'] for r in rows),
        target_evidence_used=False,
        conclusion='Both fixed pilots rejected; retain original fusion. Low-alpha overweighting is real but not a demonstrated dominant cross-benchmark bottleneck. No target evaluation or dependent-head reuse.')
    (ROOT / 'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
