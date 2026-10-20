"""Migration contracts: retain authority, grouping and typed numerical outputs."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
import torch

from radio_gs import evidence_core as core
from radio_gs.evidence_core.store import EvidenceStore
from radio_gs.querying.anchor_observation_evidence import joint_anchor_evidence
from radio_gs.querying.instance_observation_scope import InstanceObservationScope
from radio_gs.querying.response_reconstruction import AdmittedResponseSystem


def frozen_module(name):
    root = Path(__file__).resolve().parents[1]
    path = root / 'docs/history/2026-10-18-evidence-core-baseline/radio_gs/querying' / (name + '.py')
    spec = importlib.util.spec_from_file_location('radio_gs.querying._frozen_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unknown_and_inapplicable_are_not_exclusions():
    positive = np.array([True, False, False, False])
    negative = ~positive
    domains = core.compile_domains(positive, negative, visible=np.array([1, 1, 0, 1], bool),
        applicable=np.array([1, 0, 1, 1], bool), unknown=np.array([0, 0, 0, 1], bool),
        negative_origin='model_complement')
    assert domains.positive.tolist() == [True, False, False, False]
    assert not domains.negative.any()
    assert domains.unknown.tolist() == [False, True, False, True]


def test_rejected_proposal_does_not_erase_independent_positive():
    p = torch.tensor([True, False]); n = ~p; visible = torch.ones(2, dtype=torch.bool)
    veto = core.compile_domains(p, n, visible=visible, exclusion_veto=visible,
                                negative_origin='model_complement')
    unknown = core.compile_domains(p, n, visible=visible, unknown=visible,
                                   negative_origin='model_complement')
    assert veto.positive.tolist() == [True, False] and not veto.negative.any()
    assert not unknown.positive.any() and unknown.unknown.all()


def test_authority_requires_boolean_domains_disjoint_signs_and_origin():
    b = np.ones(2, bool)
    with pytest.raises(ValueError, match='both signs'):
        core.compile_domains(b, b, visible=b, negative_origin='user_annotation')
    with pytest.raises(ValueError, match='boolean'):
        core.compile_domains(b.astype(float), ~b, visible=b, negative_origin='model_complement')
    with pytest.raises(ValueError, match='origin'):
        core.compile_domains(b, ~b, visible=b, negative_origin='automatic_user_negative')


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_sparse_transport_and_complete_solver_match_frozen_code(monkeypatch, dtype):
    monkeypatch.setenv('RADIO_GS_EVIDENCE_CORE', 'shared_v1')
    a = torch.tensor([[.8, .2, 0], [.1, .9, 0], [0, 0, 1]], dtype=dtype).to_sparse()
    y = torch.tensor([[.9, .1], [.2, .8], [.5, .5]], dtype=dtype)
    known = torch.tensor([[.3, 0], [1, .8], [0, 0]], dtype=dtype)
    initial = torch.tensor([[.1, .9], [.7, .3], [.9, .2]], dtype=dtype)
    terms = [(a, y, known, 1.)]
    old = frozen_module('response_reconstruction').AdmittedResponseSystem(initial, terms)
    new = AdmittedResponseSystem(initial, terms)
    for x, z in [(new.mass, old.mass), (new.diagonal(), old.diagonal()),
                 (new.rendered(), old.rendered())]:
        assert torch.equal(x, z)
    assert torch.equal(new.rendered()[2], initial[2])
    csr = a.to_sparse_csr()
    assert torch.equal(core.project(csr, initial), torch.sparse.mm(csr, initial))
    torch.testing.assert_close(core.project(csr, initial), torch.sparse.mm(a, initial))


def relations():
    # Same-view nested masks disagree about vertex 1. Vertex 3 is unobserved.
    membership = sparse.csr_matrix(np.array([[1, 1, 1], [1, 0, 1], [0, 0, 0], [0, 0, 0]], np.float32))
    visibility = sparse.csr_matrix(np.array([[1, 1], [1, 1], [1, 0], [0, 0]], np.float32))
    return membership, visibility, np.array([0, 0, 1])


def test_grouping_retains_ambiguity_without_counting_nested_masks_as_views(monkeypatch):
    monkeypatch.setenv('RADIO_GS_EVIDENCE_CORE', 'shared_v1')
    m, v, frames = relations(); vertices = np.arange(4)
    result = joint_anchor_evidence(m, v, frames, 0, vertices)
    old = frozen_module('anchor_observation_evidence').joint_anchor_evidence(m, v, frames, 0, vertices)
    np.testing.assert_array_equal(result, old)
    np.testing.assert_array_equal(result[1], [.75, 1, .5, .5, 1, 1])
    np.testing.assert_array_equal(result[3], np.zeros(6))
    # Repeating a complete view's proposal set cannot create a new view.
    repeated = core.grouped_membership_features(m[:, [0, 1, 0, 1, 2]], v,
                                                np.array([0, 0, 0, 0, 1]), 0, vertices)
    np.testing.assert_array_equal(result, repeated)


@pytest.mark.parametrize('policy', ['visible_complement', 'covisible_completeness'])
def test_text_scope_keeps_legacy_selection_and_rank(monkeypatch, policy):
    m, v, frames = relations()
    reader = InstanceObservationScope(m, v, frames, exclusion_policy=policy)
    identity = np.array([.9, .6, .7], np.float32)
    rank = np.array([.9, .6, .3, .1], np.float32)
    monkeypatch.setenv('RADIO_GS_EVIDENCE_CORE', 'legacy'); old = reader.infer(identity, rank)
    monkeypatch.setenv('RADIO_GS_EVIDENCE_CORE', 'shared_v1'); new = reader.infer(identity, rank)
    for i in [0, 1]:
        np.testing.assert_array_equal(new[i], old[i])
    for key in old[2]:
        np.testing.assert_array_equal(new[2][key], old[2][key])
    assert old[3] == new[3]


def test_missing_anchor_and_invalid_transport_are_explicit():
    m, v, frames = relations()
    result = core.grouped_membership_features(m, v, frames, 3, np.arange(4))
    assert not result.any()
    with pytest.raises(ValueError, match='domain'):
        core.project(m, np.zeros(4))


def test_profile_typo_cannot_silently_run_legacy(monkeypatch):
    monkeypatch.setenv('RADIO_GS_EVIDENCE_CORE', 'shared_typo')
    with pytest.raises(ValueError, match='profile'):
        core.enabled()


def test_store_reads_relations_without_copy_and_checks_alignment(tmp_path):
    import json
    m, v, _ = relations()
    sparse.save_npz(tmp_path/'membership.npz', m)
    sparse.save_npz(tmp_path/'visibility.npz', v)
    np.save(tmp_path/'region_offsets.npy', [0, 2, 3])
    manifest = tmp_path/'map.json'
    manifest.write_text(json.dumps(dict(map_files=['membership.npz', 'visibility.npz', 'region_offsets.npy'])))
    before = set(tmp_path.iterdir())
    mm, vv, frames = EvidenceStore.open(manifest).registered_relations()
    assert (mm != m).nnz == 0 and (vv != v).nnz == 0
    assert frames.tolist() == [0, 0, 1] and set(tmp_path.iterdir()) == before
    np.save(tmp_path/'region_offsets.npy', [0, 1, 2])
    with pytest.raises(ValueError, match='domains'):
        EvidenceStore.open(manifest).registered_relations()
