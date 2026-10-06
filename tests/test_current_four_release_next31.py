"""Historical artifacts must be preserved exactly, never hash-blessed in place."""
import hashlib
from pathlib import Path

from scripts.audits.verify_current_four_release_next31_v2 import resolve_bound_artifact, success


def digest(value):
    return hashlib.sha256(value).hexdigest()


def test_exact_archive_resolves_drift_without_modifying_current(tmp_path):
    current = tmp_path / 'current.py'
    archive = tmp_path / 'archive.py'
    current.write_bytes(b'new implementation')
    archive.write_bytes(b'old implementation')
    row = resolve_bound_artifact('current.py', digest(b'old implementation'), root=tmp_path,
                                 archives={'current.py': 'archive.py'})
    assert success(row) and row['status'] == 'archived_match'
    assert row['original_status'] == 'mismatch'
    assert current.read_bytes() == b'new implementation'


def test_wrong_archive_never_relabels_mismatch_as_pass(tmp_path):
    (tmp_path / 'current.py').write_bytes(b'new')
    (tmp_path / 'archive.py').write_bytes(b'also wrong')
    row = resolve_bound_artifact('current.py', digest(b'old'), root=tmp_path,
                                 archives={'current.py': 'archive.py'})
    assert not success(row)


def test_unregistered_missing_file_is_not_satisfied_by_another_archive(tmp_path):
    (tmp_path / 'archive.py').write_bytes(b'old')
    row = resolve_bound_artifact('missing.py', digest(b'old'), root=tmp_path,
                                 archives={'different.py': 'archive.py'})
    assert row['status'] == 'missing' and not success(row)


def test_published_archive_bytes_match_original_independent_seals():
    root = Path(__file__).resolve().parents[1]
    expected = {
        'evaluate_full_field_rebuild_next19.py': '68dac7b79f67aae255125fb9bee7d68e7c156d92c574b369e7f9e5e0744b5a7e',
        'ludvig_run_sam_20260801.py': '1f1bc95d70f22fcfb1cfe2df1bc3f2416ebb667c2faeddb7dc24bcd9bceed68f',
    }
    for path, value in expected.items():
        assert digest((root / 'scripts/audits/frozen_sources_next31' / path).read_bytes()) == value


def test_lerf_metric_operator_is_unchanged_in_candidate_capable_entrypoint():
    root = Path(__file__).resolve().parents[1]
    old = (root / 'scripts/audits/frozen_sources_next31/evaluate_full_field_rebuild_next19.py').read_text()
    current = (root / 'scripts/audits/evaluate_full_field_rebuild_next19.py').read_text()
    def operator(source):
        return source.split(' # Render stage rejects training tensors')[1].split('\n oldinfo=')[0]
    assert operator(old) == operator(current)
