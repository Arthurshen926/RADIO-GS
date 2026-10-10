"""Apply the same post-prediction domain diagnostic to the native-prior study."""
import argparse
import torch
from scripts.research import diagnose_query_field_joint_exclusion_domain as base
from scripts.research.run_query_field_joint_exclusion_prior import ROOT,ARMS
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args()
    torch.set_num_threads(4);base.ROOT=ROOT;base.ARMS=ARMS;base.run(a.index,initial_arm='semantic_initial')
