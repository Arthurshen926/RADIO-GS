"""Independent source-denied reload of actual-member-domain coverage factors."""
import argparse
import torch
from scripts.research import validate_query_field_coverage_factors as base
from scripts.research.compile_query_field_available_coverage import ROOT,member_conditions

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True)
    p.add_argument('--memory-fraction',type=float,default=.45)
    a=p.parse_args();torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    base.ROOT=ROOT;base.member_conditions=member_conditions;base.run(a.index)
