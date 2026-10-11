"""Real-query execution parity check; no benchmark labels or scoring.

Run with --output pointing to a new JSON receipt (existing files are preserved).
"""
import argparse
import time,json,torch,numpy as np
from pathlib import Path
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.extent_decision import ExtentDecision
from radio_gs.query_field.extent_decision_linked import LinkedExtentDecision
from scripts.research.run_query_field_relations import receipt,write
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args_cli=parser.parse_args()
if args_cli.output.exists(): raise FileExistsError(args_cli.output)
torch.set_num_threads(4)
base=Path('output/query_field_20261011');scene='figurines'; qi=0
field,_=load_field(base/'compiled_extent_factors_v2'/scene/'field')
scope=torch.load(base/'extent_decision_v2/predictions/figurines__measure/privileged_scope.pt',weights_only=True)
state=torch.load(base/'canonical_subset_reference_v1/predictions/figurines__measure/admitted_optimization_state.pt',weights_only=True)[qi]
confidence=np.zeros(18)
for vi,c in zip(state['view_indices'],state['confidence']):confidence[vi]=c
args=((state['start_membership']>.5).numpy(),torch.cat([v['target'][:,qi] for v in scope['views']]).numpy(),torch.cat([v['positive'][:,qi] for v in scope['views']]).numpy(),torch.cat([v['negative'][:,qi] for v in scope['views']]).numpy(),confidence,state['support_cost'].numpy())
values={};sets={};times={}
for name,cls in [('reference',ExtentDecision),('linked',LinkedExtentDecision)]:
 decoder=cls(field.extent_factors,len(field.xyz));start=time.monotonic();selected,report=decoder.solve(*args)
 times[name]=time.monotonic()-start;values[name]=report;sets[name]=selected
 print(name,times[name],report['energy_trace'],flush=True)
 del decoder
assert np.array_equal(sets['reference'],sets['linked']) and values['reference']==values['linked']
write(args_cli.output,dict(passed=True,scene=scene,query=scope['texts'][qi],elements=len(field.xyz),reports=values,seconds=times,all_selected_bits_exact=True,all_trace_values_exact=True,benchmark_scores_opened=False,code=[receipt('radio_gs/query_field/extent_decision.py'),receipt('radio_gs/query_field/extent_decision_linked.py')]))
