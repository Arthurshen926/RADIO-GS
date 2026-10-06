"""Verify actual element and prediction changes without reading target labels."""
import json
from pathlib import Path
import cv2
import torch
from radio_gs.utils.immutable_artifacts import sha256_file

ROOT=Path('output/nvos_component_evidence_next31_v2_resume')


def main():
    summary=json.loads((ROOT/'summary.json').read_text());rows={}
    for task in summary['task_results']:
        folder=ROOT/task;r=json.loads((folder/'report.json').read_text())
        a=torch.load(folder/'baseline_elements.pt',map_location='cpu',weights_only=False)['scores']
        b=torch.load(folder/'component_evidence_elements.pt',map_location='cpu',weights_only=False)['scores']
        x=torch.load(folder/'baseline.pt',map_location='cpu',weights_only=False)['response']
        y=torch.load(folder/'component_evidence.pt',map_location='cpu',weights_only=False)['response']
        size=tuple(r['scoring_grid_wh'])
        sx=cv2.resize(x.numpy(),size,interpolation=cv2.INTER_LINEAR)>=.5
        sy=cv2.resize(y.numpy(),size,interpolation=cv2.INTER_LINEAR)>=.5
        rows[task]=dict(element_max_absolute_delta=float((a-b).abs().max()),
            changed_element_scores=int(((a-b).abs()>1e-7).sum()),
            changed_selected_elements=int(((a>=.5)!=(b>=.5)).sum()),
            response_max_absolute_delta=float((x-y).abs().max()),
            changed_prediction_grid_pixels=int(((x>=.5)!=(y>=.5)).sum()),
            changed_scoring_grid_pixels=int((sx!=sy).sum()),
            report_sha256=sha256_file(folder/'report.json'))
    original=json.loads((ROOT/'actual_output_differences.json').read_text())
    assert original['rows']==rows
    result=dict(implementation_sha256=sha256_file(__file__),
        original_diagnostic_sha256=sha256_file(ROOT/'actual_output_differences.json'),
        original_diagnostic_reproduced_exactly=True,
        candidate_is_elementwise_identity=all(r['element_max_absolute_delta']==0 for r in rows.values()),
        all_eight_candidate_scoring_masks_equal_baseline=all(r['changed_scoring_grid_pixels']==0 for r in rows.values()),rows=rows)
    out=ROOT/'actual_output_differences_verified.json';assert not out.exists()
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
