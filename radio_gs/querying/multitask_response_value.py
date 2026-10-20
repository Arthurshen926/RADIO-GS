"""Shared utility supervision and multi-output guard for intact category menus.

Task labels define balancing and calibration only, never estimator inputs.
Expanding menu metric guards repeats the SAME gains; it cannot choose a different
answer for an easier class subset.
"""
import numpy as np
from .response_value import relative_utilities


def category_report(report, *, fit):
    rows = report['rows'] if fit else [r for r in report['rows'] if r['fold'] == 'full']
    if len(rows) != (4 if fit else 1) or sum(r['fold'] == 'full' for r in rows) != 1:
        raise ValueError('fixed full-map and three-fold category episodes required')
    result = []
    for row in rows:
        metrics = row['menu_utilities']
        if set(metrics) != {'19', '15', '10'}:
            raise ValueError('all menu utility outputs must be retained')
        values = np.asarray([metrics[k] for k in ('19','15','10')], dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(row['features']) or not np.isfinite(values).all():
            raise ValueError('one aligned utility vector per complete menu')
        result.append(dict(row, ious=values.mean(0).tolist(), modality='category'))
    return dict(report, rows=result, modality='category', queries=len(result))


def predict(reports, model, *, blind=False, permuted=False):
    return [dict(scene=r['scene'], modality=r['modality'], query_index=row['query_index'], ious=row['ious'],
        gains=relative_utilities(model, row['permuted_features' if permuted else 'features'], query_blind=blind).tolist(),
        menu_utilities=row.get('menu_utilities'), evidence_available=row['evidence']['verification_views'] > 0)
        for r in reports for row in r['rows']]


def protected_output_rows(rows):
    result = []
    for row in rows:
        outputs = row['menu_utilities'] if row['modality'] == 'category' else {row['modality']: row['ious']}
        for name, utility in outputs.items():
            if len(utility) != len(row['gains']):
                raise ValueError('menu metric changed the complete candidate order')
            result.append(dict(row, scene=row['scene']+'|'+row['modality']+'|'+name, ious=utility))
    return result
