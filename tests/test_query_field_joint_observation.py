import numpy as np
import pytest
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier
import torch

from radio_gs.query_field.compiled_forest import compile_forest, CompiledForest
from radio_gs.query_field.compile_joint_observation import observation_state
from radio_gs.query_field.joint_observation import JointObservationEvidence, enable
from radio_gs.query_field.contracts import Support, QueryPacket
from radio_gs.query_field.storage import save_field, load_field
from radio_gs.query_field.runtime import QueryFieldRuntime
from radio_gs.querying.direct_anchor_membership import DirectAnchorEvidence
from radio_gs.querying.anchor_observation_evidence import joint_anchor_evidence
from test_query_preserving_field import example


def fixture():
    rng=np.random.default_rng(12)
    frames=np.array([0,0,1,1,2])
    b=sparse.csr_matrix(rng.integers(0,2,(12,5)).astype(np.float32))
    v=sparse.csr_matrix(np.column_stack([np.asarray(b[:,frames==f].max(1).toarray()).ravel() for f in range(3)]))
    # An explicitly visible exclusion, and an entirely unobserved row.
    b=b.tolil();v=v.tolil(); b[10,:]=0;v[10,:]=1;b[11,:]=0;v[11,:]=0
    b=b.tocsr();v=v.tocsr(); b.eliminate_zeros();v.eliminate_zeros()
    descriptors=rng.normal(size=(5,7)).astype(np.float32)
    state,transfer=observation_state(b,v,frames,descriptors)
    field=example();field.world_units_per_metre=1.
    reference=DirectAnchorEvidence(b,v,frames,field.xyz.numpy(),descriptors,transfer)
    return field,state,reference


def classifier(dimensions=15):
    rng=np.random.default_rng(71);x=rng.normal(size=(1200,dimensions)).astype(np.float32)
    y=x[:,0]-.2*x[:,1]>.1
    model=ExtraTreesClassifier(n_estimators=7,max_leaf_nodes=31,random_state=8).fit(x,y)
    return model,x


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_tensor_forest_preserves_real_operator_and_float32_threshold_boundaries(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    model,x=classifier();edges=[]
    for estimator in model.estimators_:
        tree=estimator.tree_
        for feature,threshold in zip(tree.feature,tree.threshold):
            if feature<0:continue
            for value in [np.nextafter(np.float32(threshold),np.float32(-np.inf)),np.float32(threshold),
                          np.nextafter(np.float32(threshold),np.float32(np.inf))]:
                row=np.zeros(15,np.float32);row[feature]=value;edges.append(row)
    x=np.concatenate([x,np.asarray(edges)])
    forest=CompiledForest(compile_forest(model,.35)).to(device)
    result=forest.probability(torch.from_numpy(x).to(device),chunk_size=127).cpu().numpy()
    assert np.array_equal(result,model.predict_proba(x)[:,1].astype(np.float32))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_complete_anchor_functions_match_v10_before_member_decision(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    field,state,reference=fixture();field=field.to(device)
    relation=JointObservationEvidence(state,12).to(device)
    for anchor in range(12):
        support=Support(torch.tensor([anchor]),torch.ones(1))
        result=relation.features(field,support).cpu()
        expected=reference.encode(anchor,np.arange(12))
        torch.testing.assert_close(result,torch.from_numpy(expected),atol=1e-6,rtol=1e-5)
        rows=torch.tensor([10,3,11,0])
        torch.testing.assert_close(relation.features(field,support,rows.to(device)).cpu(),result[rows],atol=1e-6,rtol=1e-5)


def test_unknown_dilution_and_repeated_supports_keep_their_input_mass():
    field,state,_=fixture();relation=JointObservationEvidence(state,12)
    one=relation.joint_features(Support(torch.tensor([0]),torch.ones(1)))
    # Unknown support has no observation weight; visibility fractions still
    # distinguish missing evidence from an observed exclusion.
    mix=relation.joint_features(Support(torch.tensor([0,11]),torch.tensor([.25,.75])))
    torch.testing.assert_close(mix,one)
    repeated=relation.joint_features(Support(torch.tensor([0,0,11]),torch.tensor([.1,.15,.75])))
    torch.testing.assert_close(mix,repeated)
    assert not relation.joint_features(Support(torch.tensor([11]),torch.ones(1))).any()
    assert mix[10,5] == 1 and mix[11,5] == 0


def test_native_moment_must_retain_its_magnitude():
    field,state,reference=fixture();relation=JointObservationEvidence(state,12)
    actual=relation.features(field,Support(torch.tensor([0]),torch.ones(1)))[:,13]
    expected=reference.encode(0,np.arange(12))[:,13]
    torch.testing.assert_close(actual,torch.from_numpy(expected))
    moment=state['native_moment'];normalized=torch.nn.functional.normalize(moment,dim=1)
    assert (actual-normalized@normalized[0]).abs().max()>.01


@pytest.mark.parametrize('supervised',[False,True])
def test_export_all_inputs_share_complete_carrier_with_separate_native_support(tmp_path,supervised):
    field,state,_=fixture();field.observed[11]=False
    if supervised:
        model,_=classifier();state.update({'forest.'+k:v for k,v in compile_forest(model,.35).items()})
    enable(field,state)
    save_field(field,tmp_path/'field',dict(additional_3d_labels=supervised))
    loaded,manifest=load_field(tmp_path/'field');runtime=QueryFieldRuntime(loaded)
    assert not loaded.observed[11] and loaded.member_available[11]
    # Registered geometric support is valid even without native observations.
    packets=[QueryPacket('point_3d',positive=Support(torch.tensor([11]),torch.ones(1)),exact_positive=True),
        QueryPacket('point_2d',positive=Support(torch.tensor([0,1]),torch.tensor([.2,.8])),
            negative=Support(torch.tensor([2]),torch.ones(1))),
        QueryPacket('text',semantic=field.semantic.decode(torch.tensor([0]))[0]),
        QueryPacket('image',semantic=field.semantic.decode(torch.tensor([0]))[0],local=field.local.decode(torch.tensor([0])))]
    for packet in packets:
        a=QueryFieldRuntime(field).query(packet);b=runtime.query(packet)
        assert torch.equal(a.membership_scores,b.membership_scores)
        assert torch.equal(a.selected,b.selected)
    assert runtime.query(packets[0]).selected[11]
    assert not manifest['external_scene_dependencies']


def test_reject_membership_without_visibility():
    b=sparse.csr_matrix([[1]],dtype=np.float32);v=sparse.csr_matrix([[0]],dtype=np.float32)
    with pytest.raises(ValueError,match='outside declared visibility'):
        observation_state(b,v,np.array([0]),np.ones((1,2),np.float32))
