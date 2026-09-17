import torch
from radio_gs.models.extent_benefit_gate import ExtentUtilityTree
from radio_gs.training.extent_utility_tree import fit_extent_utility_tree


def test_utility_policy_separates_harmful_overrides_and_preserves_ties():
    x=torch.zeros(20,8);x[:,0]=torch.arange(20).float()
    gain=torch.cat([torch.full((10,),-.2),torch.full((10,),.5)])
    tree=ExtentUtilityTree();tree.load_state_dict(fit_extent_utility_tree(x,gain,torch.ones(20),torch.arange(20)))
    expected=gain>0;assert torch.equal(tree.decide_features(x),expected)
    assert torch.equal(tree.decide_features(x[:3]),expected[:3])
    clone=ExtentUtilityTree();clone.load_state_dict(tree.state_dict());assert torch.equal(clone.decide_features(x),expected)
    tree.load_state_dict(fit_extent_utility_tree(x,torch.zeros(20),torch.ones(20),torch.arange(20)))
    assert not tree.decide_features(x).any()


def test_duplicate_query_variants_cannot_satisfy_minimum_group_requirement():
    x=torch.zeros(20,8);x[:,0]=torch.arange(20).float();gain=torch.arange(20).float()-10
    tree=fit_extent_utility_tree(x,gain,torch.ones(20),torch.zeros(20,dtype=torch.long))
    assert (tree['feature']==-1).all()
