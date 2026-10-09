"""Replay all eight authorized NVOS prompts and audit relation admission."""
import gc
from pathlib import Path
from types import SimpleNamespace

import torch

from radio_gs.query_field.contracts import Support
from radio_gs.query_field.geometry import projection
from radio_gs.query_field.public_nvos import load_strokes
from radio_gs.query_field.storage import load_field
from scripts.research.run_query_field_mixture_functions import ROOT, PREVIOUS, ARMS, read, write, receipt, check
from scripts.research.diagnose_query_field_support_composition import association_variants
from scripts.research.run_query_field_protected import ROOT as PROTECTED


@torch.no_grad()
def main():
    from radio_gs.query_field.adapters import scribble_packet, raster_extent_scale
    read(ROOT / 'scoring_complete.json')
    cohort = read(ROOT / ARMS[0] / 'cohort.json')
    out = ROOT / 'diagnostics/signed_activation'; out.mkdir(parents=True, exist_ok=False)
    records = []
    for item in cohort['nvos']:
        name = item['name']; path = ROOT / ARMS[0] / 'predictions/nvos' / name / name
        prediction = read(path / 'prediction.json'); config = prediction['configuration']; job = prediction['job']
        check(dict(path=str(path / 'predictions.pt'), sha256=prediction['predictions_sha256']))
        field, _ = load_field(job['field'], 'cuda')
        fixture = torch.load(job['camera_fixture'], map_location='cpu', weights_only=True)
        reference = SimpleNamespace(**fixture['reference']); height, width = config['projection_hw']
        masks, _ = load_strokes(job['positive'], job['negative'], (height, width))
        geometry = SimpleNamespace(xyz=field.xyz, get_xyz=lambda: field.xyz,
            get_rotation=lambda: field.quaternions, get_scaling=lambda: field.scales, get_opacity=lambda: field.opacities)
        _, operator, _, _ = projection(geometry, reference, height, width, fixture['world_units_per_metre'])
        scale = None if job.get('scope', 'default') == 'default' else raster_extent_scale(
            operator, masks[0], field.xyz, fixture['world_units_per_metre'], field.default_scale)
        packet, _ = scribble_packet(operator, masks[0], masks[1] if job['mode'] == 'signed' else None,
            scale=scale, preserve_rays=field.support_aggregation == 'expected_energy',
            negative_rays=field.negative_rule == 'competitive_rays')
        actual = field.query(packet); saved = torch.load(path / 'predictions.pt', map_location='cpu', weights_only=True)
        error = float((actual.membership_scores.cpu() - saved['membership']).abs().max())
        if error > 1e-6 or not torch.equal(actual.selected.cpu(), saved['selected']):
            raise ValueError('authorized prompt replay changed prediction')
        prior_path = PROTECTED / 'protected_training/predictions/nvos' / name / name
        prior_receipt = read(prior_path / 'prediction.json')
        check(dict(path=str(prior_path / 'predictions.pt'), sha256=prior_receipt['predictions_sha256']))
        prior = torch.load(prior_path / 'predictions.pt', map_location='cpu', weights_only=True)
        unchanged = {key: torch.equal(saved[key], prior[key]) for key in saved}
        soft_path = PREVIOUS / 'soft_paired_function/predictions/nvos' / name / name
        soft_receipt = read(soft_path / 'prediction.json')
        check(dict(path=str(soft_path / 'predictions.pt'), sha256=soft_receipt['predictions_sha256']))
        soft = torch.load(soft_path / 'predictions.pt', map_location='cpu', weights_only=True)
        soft_unchanged = {key: torch.equal(saved[key], soft[key]) for key in saved}
        supports = {}
        for sign in ['positive', 'negative']:
            support = getattr(packet, sign)
            if support is None:
                supports[sign] = None; continue
            mask = field.observed[support.indices] & (support.weights > 0)
            ids, weights = support.indices[mask], support.weights[mask]
            if not len(ids):
                supports[sign] = dict(observed_support_rows=0, admitted_views=0); continue
            support = Support(ids, weights); relation = field.evidence
            association = torch.sparse.mm(relation.matrix('association', ids).transpose(0, 1),
                (weights / weights.sum())[:, None]).flatten()
            coefficient = relation.coefficients(support, actual.scale)
            pos, neg = relation.masses(support, actual.scale)
            supports[sign] = dict(composition=association_variants(relation, support, actual.scale),
                observed_support_rows=len(ids), maximum_association=float(association.max()),
                association_coordinates_at_majority=int((association >= .5).sum()),
                admitted_coordinates=int((coefficient > 0).sum()),
                admitted_views=int(torch.unique(relation.views[coefficient > 0]).numel()),
                effective_view_mass=float(coefficient.sum()),
                activated_observed_rows=int((((pos + neg) > 0) & field.observed).sum()))
        records.append(dict(query=name, supports=supports, scale=actual.scale, prediction_replay_max_error=error,
            exact_parent_outputs=unchanged, exact_soft_outputs=soft_unchanged, parent_prediction=receipt(prior_path / 'prediction.json'),
            prediction=receipt(path / 'prediction.json'), fixture=receipt(job['camera_fixture']),
            positive=receipt(job['positive']), negative=receipt(job['negative'])))
        print('signed admission audit', name, supports, flush=True)
        del actual, field, operator, relation, geometry, packet; gc.collect(); torch.cuda.empty_cache()
    if len(records) != 8: raise ValueError('incomplete signed prompt cohort')
    write(out / 'report.json', dict(records=records, code=receipt(Path(__file__)), source_access=False,
        current_authorized_prompts_only=True, target_labels_accessed=False, no_model_change=True,
        max_prediction_error=max(r['prediction_replay_max_error'] for r in records)))


if __name__ == '__main__':
    torch.set_num_threads(4); main()
