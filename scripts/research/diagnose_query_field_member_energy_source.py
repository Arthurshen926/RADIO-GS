"""Full source extent diagnostic AFTER predictions; no changes to inference."""
import argparse
import torch
from scripts.research import diagnose_query_field_joint_exclusion_source as base
from scripts.research.run_query_field_member_energy import ROOT,ARMS
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.60)
    base.ROOT=ROOT;base.ARMS=ARMS;base.run(a.index,initial_arm='semantic_initial')
