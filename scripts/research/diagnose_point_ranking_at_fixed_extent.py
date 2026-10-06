"""One fixed-size readout control; no target masks or fitted cutoffs are inputs.

This reuses this experiment's frozen model ranks and freshly computed anchor
co-membership. It diagnoses ranking at the original geometric selection size,
and cannot demonstrate complete-instance size prediction.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--system',type=Path,required=True)
    p.add_argument('--package',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    prediction=a.output/'predictions';selection=a.output/'selections'
    prediction.mkdir(parents=True,exist_ok=True);selection.mkdir(exist_ok=True)
    source=a.system/'uqis';rows=[]
    for modality in ['text','image','point_2d','point_3d']:
        queries=json.loads((a.package/'method/queries'/f'{modality}.json').read_text())['queries']
        for q in queries:
            name=q['query_id']+'.npy';rank=np.load(source/'identity_scope/predictions'/name)
            os.link(source/'identity_scope/predictions'/name,prediction/name)
            if modality in ['point_2d','point_3d']:
                anchor=np.load(source/'anchor_comembership/predictions'/name);k=int((anchor>.5).sum())
                order=np.lexsort((np.arange(len(rank)),-anchor,-rank))
                mask=np.zeros(len(rank),bool);mask[order[:k]]=True
                np.save(selection/name,mask)
                rows.append(dict(query_id=q['query_id'],modality=modality,source_count=k,
                    selected_count=int(mask.sum()),cutoff_tie=bool(0<k<len(rank) and rank[order[k-1]]==rank[order[k]])))
            else:os.link(source/'identity_scope/selections'/name,selection/name)
    (a.output/'method.json').write_text(json.dumps(dict(current_model_ranks=str(source/'identity_scope/predictions'),
        current_geometric_extent=str(source/'anchor_comembership/predictions'),label_access_before_selection=False,
        controls='all model ranks unchanged; point selection count from same-run anchor geometry',
        tie_break='original anchor support, then stable vertex index',
        not_a_solution_for_complete_instance_size=True,rows=rows),indent=2)+'\n')
    subprocess.run([sys.executable,'-m','radio_gs.benchmarks.uqis.evaluate','--package',str(a.package),
        '--predictions',str(prediction),'--selections',str(selection),'--output',str(a.output/'result.json')],check=True)


if __name__=='__main__':main()
