"""Execution-equivalent engine for the frozen full-extent member policy."""
import argparse
from pathlib import Path
import torch
from radio_gs.query_field.extent_decision_linked import LinkedExtentDecision
from scripts.research import run_query_field_extent_decision as base

base.ROOT = Path('output/query_field_20261011/extent_decision_v3')
base.PROTOCOL = Path('docs/experiments/2026-10-11-extent-decision-v3-protocol.md')
base.CODE = [*base.CODE, __file__, 'radio_gs/query_field/extent_decision_linked.py']
base.ExtentDecision = LinkedExtentDecision


def run():
    import time
    from concurrent.futures import ThreadPoolExecutor
    from scripts.research.run_query_field_relations import read, execute
    cfg = read(base.ROOT/'prediction_config.json')
    def predict(i):
        execute(['-m','scripts.research.run_query_field_extent_linked','predict','--index',str(i)],
                base.ROOT/'logs'/f'predict_{i}.log', [0,2,0,2][i])
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(predict,i) for i in range(2)]
        while not all((base.ROOT/'predictions'/j['name']/'privileged_scope.pt').exists() for j in cfg['jobs'][:2]):
            for f in jobs:
                if f.done(): f.result()
            time.sleep(1)
        time.sleep(3)
        jobs += [pool.submit(predict,i) for i in [2,3]]
        for f in jobs: f.result()
    execute(['-m','scripts.research.run_query_field_extent_linked','complete'],base.ROOT/'complete.log')
    execute(['-m','scripts.research.run_query_field_extent_linked','audit'],base.ROOT/'audit.log')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('stage', choices=['prepare','predict','complete','audit','run']);p.add_argument('--index',type=int)
    a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='prepare':base.prepare()
    elif a.stage=='predict':base.predict(a.index)
    elif a.stage=='run':run()
    elif a.stage=='complete':
        from scripts.research.complete_query_field_extent_decision import main
        main()
    else:
        from scripts.research.audit_query_field_extent_decision import main
        main()
