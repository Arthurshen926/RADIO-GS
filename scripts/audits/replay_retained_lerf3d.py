"""Replay sealed historical LERF3D arguments with current baseline code.

This is an explicit adapter outside v4. It reuses historical query score caches,
not v4 predictions, and exports masks for independent scoring. A successful run
is a renderer replay, not a reconstruction/teacher retraining claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

source, destination = map(Path, sys.argv[1:3])
module = sys.argv[3] if len(sys.argv) > 3 else 'radio_gs.scripts.eval_lerf_direct_3d_selection'
if module not in {'radio_gs.scripts.eval_lerf_direct_3d_selection', 'radio_gs.scripts.eval_lerf_grounding'}:
    raise ValueError('unsupported baseline evaluator')
record = json.loads(source.read_text())
historical = record['args']
original_parse = argparse.ArgumentParser.parse_args

def parse(parser, args=None, namespace=None):
    arguments = []
    actions = {a.dest: a for a in parser._actions}
    unknown = set(historical) - set(actions)
    if unknown:
        raise ValueError('historical arguments absent from current parser: ' + str(sorted(unknown)))
    overrides = {'output_dir': str(destination)}
    if 'save_masks' in actions:
        overrides['save_masks'] = True
    for key, value in {**historical, **overrides}.items():
        action = actions[key]
        option = next((x for x in action.option_strings if x.startswith('--')), None)
        if option is None:
            raise ValueError('cannot replay positional argument: ' + key)
        if isinstance(action, argparse._StoreTrueAction):
            if value is True or value == 'True':
                arguments.append(option)
        elif isinstance(action, argparse._StoreFalseAction):
            if value is False or value == 'False':
                arguments.append(option)
        else:
            if value is None or value == 'None':
                if action.required:
                    raise ValueError('required historical argument is None: '+key)
                if action.default is not None:
                    raise ValueError('historical None differs from current default: '+key)
                continue
            arguments.extend([option, str(value)])
    destination.mkdir(parents=True, exist_ok=True)
    receipt = {'source_result': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'replay_role': 'historical_score_cache_renderer_replay', 'argv': arguments}
    with (destination/'replay_receipt.json').open('x') as f:
        json.dump(receipt, f, indent=2)
    return original_parse(parser, arguments, namespace)

argparse.ArgumentParser.parse_args = parse
sys.argv = ['eval_lerf_direct_3d_selection']
runpy.run_module(module, run_name='__main__')
