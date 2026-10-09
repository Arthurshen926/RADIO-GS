from pathlib import Path
import json
import numpy as np
from scipy import sparse
from radio_gs.querying.direct_anchor_membership import view_balanced_transfer
from scripts.research.run_query_field_joint_observation import ROOT,read,write,receipt
records=[]
for j in read(ROOT/'design.json')['fields']:
 p=Path(j['legacy_map']).parent
 b=sparse.load_npz(p/'membership.npz');t=sparse.load_npz(p/'transfer.npz')
 expected=view_balanced_transfer(b,np.load(p/'region_offsets.npy')).tocsr();difference=(t-expected).tocsr();difference.eliminate_zeros()
 maximum=float(np.abs(difference.data).max()) if difference.nnz else 0.
 record=dict(scene=j['scene'],maximum_absolute_error=maximum,different_entries=difference.nnz,
  transfer=receipt(p/'transfer.npz'),membership=receipt(p/'membership.npz'))
 records.append(record);print(j['scene'],maximum,difference.nnz,flush=True)
 if maximum>1e-7:raise ValueError('cached v10 transfer differs from declared native-response function')
write(ROOT/'diagnostics/legacy_native_transfer.json',dict(records=records,code=receipt(Path(__file__)),
 benchmark_queries_used=False,benchmark_labels_used=False,model_change=False,
 interpretation='all-source cached v10 operator matches declared view-balanced native function; budget subsets recompute this same function'))
