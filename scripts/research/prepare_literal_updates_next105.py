"""Source-only replacement: new pair must beat prior exact-query verifier margin.
Unchanged or unqualified proposals keep the retained next104 output.
"""
import argparse,json,shutil
from pathlib import Path
import torch
load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();key='lerf_'+a.scene;new=json.loads((Path('output/literal_seeds_next105')/key/'proposals.json').read_text());old={r['query_index']:r for r in json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text())};out=Path('output/literal_update_tracks_next105')/key;out.mkdir(exist_ok=True,parents=True);report=[]
for row in new:
 qi=row['query_index'];s=load(Path('output/literal_seeds_next105')/key/f'{qi}.pt');b=load(Path('output/grounded_seeds_next88')/key/f'{qi}.pt');same=set(s['pair_nodes'])==set(b['pair_nodes']);qualifies=not same and not row['native_fallback'] and row['selected_margin']>old[qi]['selected_margin'];report.append(dict(query=row['query'],query_index=qi,update=qualifies,same_pair=same,old_margin=old[qi]['selected_margin'],new_margin=row['selected_margin']))
 if not qualifies:shutil.copyfile(Path('output/core_tracks_next104')/key/f'{qi}.pt',out/f'{qi}.pt')
root=Path('output/literal_update_next105');root.mkdir(exist_ok=True,parents=True);(root/(key+'_decisions.json')).write_text(json.dumps(report,indent=2)+'\n');print(report)
