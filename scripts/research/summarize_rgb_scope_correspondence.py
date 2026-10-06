"""Summarize the bounded independent-RGB-correspondence experiment."""
import json
from pathlib import Path

ROOT = Path('output/rgb_scope_correspondence')


def read(path):
    return json.loads(Path(path).read_text())


def lerf_metrics(path):
    value = read(path)['metrics']['hierarchical_pipeline']
    return {**{key: 100 * value[key] for key in ['iou', 'ap', 'localization']}, 'selected_projection_iou': 100 * value['selected']['miou']}


def reference_metrics(path):
    value = read(path)['metrics']
    return {'iou': 100 * value['iou'], 'ap': 100 * value['average_precision']}


rows = []
for scene in ['ramen', 'teatime']:
    baseline = lerf_metrics(Path('output/consolidated_response_control') / scene / 'fusion/native_protected/evaluation/evaluation.json')
    candidate = lerf_metrics(ROOT / scene / 'evaluation/evaluation.json')
    diagnostics = read(ROOT / scene / 'summary.json')
    rows.append(dict(scene=scene, benchmark='LERF_development', baseline=baseline, candidate=candidate,
                     delta_pp={key: candidate[key] - value for key, value in baseline.items()},
                     diagnostics={key: diagnostics[key] for key in ['source_images', 'pairs', 'raw_matches', 'admitted_matches', 'matched_pixels', 'scope_negative_conflict_pixels', 'gaussian_reprojection_median_raster_pixels', 'gaussian_reprojection_p90_raster_pixels', 'seconds']}))
for scene in ['trex', 'orchids']:
    baseline = reference_metrics(Path('output/reference_integration/replay') / scene / 'evaluation/evaluation.json')
    candidate = reference_metrics(ROOT / ('nvos_' + scene) / 'evaluation/evaluation.json')
    diagnostics = read(ROOT / ('nvos_' + scene) / 'summary.json')
    rows.append(dict(scene=scene, benchmark='NVOS_positive_only_reference_development', baseline=baseline, candidate=candidate,
                     delta_pp={key: candidate[key] - value for key, value in baseline.items()},
                     diagnostics={key: diagnostics[key] for key in ['source_images', 'pairs', 'raw_matches', 'admitted_matches', 'matched_pixels', 'scope_negative_conflict_pixels', 'gaussian_reprojection_median_raster_pixels', 'gaussian_reprojection_p90_raster_pixels', 'seconds']}))
result = dict(
    decision='not_promoted_close_sparse_negative_conflict_admission_direction',
    reports=4,
    baseline_unchanged=True,
    method='LoFTR source-RGB correspondences independent of Gaussian maps; registered-camera fine-window epipolar constraint; only matched negative versus other positive scope pixels become unknown',
    candidate_is_not_a_new_complete_method=True,
    limitations=['LERF values are two repeatedly used development scenes, not full four-scene or unseen-scene results',
                 'NVOS covers two positive-only reference queries, response output only; no new selected-set result',
                 'LoFTR and calibrated cameras provide correspondence evidence, not independent geometry or identity ground truth',
                 'Matched points favour visible textured areas, so sparse agreement cannot certify complete mask scope or object identity',
                 'Gaussian centroid reprojection combines primitive footprint mixing, frozen-geometry error, image discretization and RGB-matcher error'],
    next_mechanism=dict(
        name='RGB_correspondence_conditioned_pixel_to_primitive_responsibility',
        proposal='For a verified RGB ray pair (a,p)<->(b,q), replace per-pixel write responsibility by normalized product of their same-Gaussian visible contributions only where common support exists; preserve original writing outside supported RGB correspondences.',
        distinction='Independent RGB decides which pixels correspond; Gaussian overlap only assigns those external correspondences to map elements. This differs from using the map itself to invent correspondences or merely deleting conflict labels.',
        necessary_control='Before fitting query fields, compare existing versus paired-ray responsibilities for source-only third-view reprojection, using a third RGB match absent from the responsibility construction. Measure shared-support coverage, descriptor retention and boundary spill. Do not tune on final GT.',
        status='proposal_only_not_implemented_or_claimed_effective',
        residual_risk='If no frozen Gaussian jointly supports both rays, the issue may be carrier/pose representation rather than assignment; do not force matches or invent negative labels. Paired overlaps can prefer occluded/wrong repeated structures and must be independently checked.'),
    results=rows,
)
(ROOT / 'conclusion.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: result[key] for key in ['decision', 'reports', 'results']}, indent=2))
