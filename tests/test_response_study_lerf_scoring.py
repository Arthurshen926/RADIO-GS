"""The response-only study must retain semantic IoU and measure harmful edits."""
from types import SimpleNamespace
import numpy as np
import torch


def test_response_only_scoring_retains_iou_and_counts_corrections(monkeypatch,tmp_path):
    import gsplat
    import radio_gs.field.moge_geometry_prior as cameras
    import radio_gs.scripts.eval_prerendered_lerf_features as scoring
    from radio_gs.evaluation.lerf_readouts import evaluate_lerf_readouts
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self,*a,**k:self)
    camera=SimpleNamespace(intrinsic=torch.eye(3),width=2,height=2,world_to_camera=torch.eye(4))
    monkeypatch.setattr(cameras,'MogeGeometryPrior',lambda *a:SimpleNamespace(cameras={1:camera},scale=1.))
    obj=SimpleNamespace(mask=np.array([[True,False],[False,False]]),query='object',bboxes=[])
    monkeypatch.setattr(scoring,'load_lerf_objects',lambda *a,**k:{'frame_00001':[obj]})
    monkeypatch.setattr(scoring,'_localization_hit',lambda *a:True)
    def rasterization(**kwargs):
        return kwargs['colors'].reshape(1,2,2,2),torch.ones(1,2,2,1),{}
    monkeypatch.setattr(gsplat,'rasterization',rasterization)
    geo=SimpleNamespace(xyz=torch.zeros(4,3),quaternions=torch.zeros(4,4),scales=torch.ones(4,3),opacities=torch.ones(4))
    observed=torch.ones(4,dtype=torch.bool); records={}
    # Reference is intentionally not first: comparison must use the named arm.
    scores={'candidate':torch.tensor([[.9],[.2],[.8],[.1]]),'baseline':torch.tensor([[.9],[.6],[.2],[.1]])}
    result=evaluate_lerf_readouts(scene='ramen',out=tmp_path,texts=['object'],scores=scores,
        selection_scores=None,geo=geo,default_observed=observed,semantic_geo=geo,semantic_observed=observed,
        selection_geo=None,selection_observed=None,camera_report='unused',camera_report_sha256='unused',
        score_selected=False,comparison_reference='baseline',report_writer=lambda name,data:records.update(data))
    assert result['baseline']['iou']==.5 and result['candidate']['iou']==.5
    assert 'selected' not in result['candidate']
    assert records['candidate'][0]['corrected_pixels']==1
    assert records['candidate'][0]['harmed_pixels']==1
    assert not list(tmp_path.iterdir())
