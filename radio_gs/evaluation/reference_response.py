"""NVOS response diagnostics; GT conventions and image access remain explicit."""
import cv2
import numpy as np
import torch

from radio_gs.training.retrieval_metrics import binary_ranking_metrics


def readout_permissions(readout, evaluation, *, selected_subset=False):
    metadata=readout.get('reference_completion')
    if metadata is None:
        return dict(target_rgb=False,negative_scribble=False,target_gt=False)
    from radio_gs.pipeline.reference_completion_contract import PERMISSIONS
    if (metadata['permissions']!=PERMISSIONS or metadata['target_frame']!=evaluation['target_frame']
            or metadata['native_wh']!=evaluation['raster_wh'] or metadata['threshold']!=evaluation['threshold']):
        raise ValueError('reference completion readout/evaluation contract differs')
    if selected_subset:
        if metadata['selection_profile']!='inherited_source_only_selected_set_unchanged':
            raise ValueError('unknown reference completion selection lineage')
        return dict(target_rgb=False,negative_scribble=False,target_gt=False)
    return dict(metadata['permissions'])


def score_response_tracks(response, truth, threshold=.5):
    raw=np.asarray(truth)
    if raw.ndim!=2 or raw.max()==0:
        raise ValueError('both frozen NVOS GT conventions require a nonempty grayscale mask')
    value=np.asarray(response,dtype=np.float32)
    if value.ndim!=2 or not np.isfinite(value).all() or value.min()<0 or value.max()>1+1e-6:
        raise ValueError('invalid continuous reference response')
    value=cv2.resize(value,(raw.shape[1],raw.shape[0]),interpolation=cv2.INTER_LINEAR)
    binary=value>=threshold
    rows={}
    for track,target in [('retained_gt_gt127',raw>127),('ludvig_gt_maximum',(raw//raw.max()).astype(bool))]:
        tn,fp,fn,tp=np.bincount((2*target.astype(np.uint8)+binary.astype(np.uint8)).ravel(),minlength=4).tolist()
        rows[track]=dict(tp=tp,fp=fp,fn=fn,tn=tn,iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
            precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,
            accuracy=(tp+tn)/target.size,
            **binary_ranking_metrics(torch.from_numpy(value).flatten(),torch.from_numpy(target).flatten()))
    return rows
