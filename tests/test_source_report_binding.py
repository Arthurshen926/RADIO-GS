import json
import pytest
from radio_gs.geometry.source_report_binding import load_source_geometry_report
from radio_gs.utils.immutable_artifacts import sha256_file


def fixture(root, bundle='geometry-digest'):
    report = root/'geometry_report.json'
    report.write_text(json.dumps({'results': {'independent_scale': {'geometry_sha256': bundle}}}))
    authority = root/'responsibility.json'
    authority.write_text(json.dumps({'metadata': {
        'frozen_geometry_report_sha256': sha256_file(report),
        'frozen_geometry_sha256': 'geometry-digest'}}))
    (root/'teachers.json').write_text(json.dumps({'authority_sha256': sha256_file(authority)}))
    return report


def test_explicit_report_must_match_both_authority_identities(tmp_path):
    report = fixture(tmp_path)
    path, data = load_source_geometry_report(tmp_path, 'arbitrary-scene', report)
    assert path == report
    assert data['results']['independent_scale']['geometry_sha256'] == 'geometry-digest'


def test_changed_report_is_rejected(tmp_path):
    report = fixture(tmp_path)
    report.write_text(report.read_text()+' ')
    with pytest.raises(ValueError, match='report differs'):
        load_source_geometry_report(tmp_path, 'scene', report)


def test_changed_authority_is_rejected(tmp_path):
    report = fixture(tmp_path)
    authority = tmp_path/'responsibility.json'
    authority.write_text(authority.read_text()+' ')
    with pytest.raises(ValueError, match='authority changed'):
        load_source_geometry_report(tmp_path, 'scene', report)


def test_report_with_wrong_bundle_is_rejected(tmp_path):
    report = fixture(tmp_path, 'other-geometry')
    with pytest.raises(ValueError, match='bundle differs'):
        load_source_geometry_report(tmp_path, 'scene', report)
