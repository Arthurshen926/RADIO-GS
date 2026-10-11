"""Repeat independent nonlinear-coverage gates on the expanded member domain."""
import argparse
import torch
from scripts.research import validate_query_field_joint_exclusion as base
from scripts.research.compile_query_field_available_coverage import ROOT as FIELDS
from scripts.research.run_query_field_joint_exclusion_domain import ROOT

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.60)
    base.ROOT=ROOT;base.FIELDS=FIELDS;base.run(a.index)
