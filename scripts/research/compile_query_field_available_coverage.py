"""Matched coverage-domain correction; source labels and sample positions unchanged."""
import argparse
from pathlib import Path
import torch
from scripts.research import compile_query_field_coverage_factors as base

ROOT=Path('output/query_field_20261011/compiled_available_coverage_v1')
PROTOCOL=Path('docs/experiments/2026-10-11-available-coverage-protocol.md')
original_conditions=base.member_conditions


def member_conditions(field, vi, back):
    cases=original_conditions(field,vi,back)
    available=field.observed|field.semantic_measure.supported
    return cases[:-1]+[('semantic_only',(available&~field.observed).float()),
                       ('unavailable',(~available).float())]+cases[-1:]


def run(index):
    base.ROOT=ROOT; base.PROTOCOL=PROTOCOL; base.member_conditions=member_conditions
    base.run(index,member_domain_name='available')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True)
    p.add_argument('--memory-fraction',type=float,default=.45)
    a=p.parse_args();torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    run(a.index)
