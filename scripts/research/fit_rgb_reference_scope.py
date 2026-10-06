"""Thin adapter from an explicit evidence manifest to shared reference fitting."""
import argparse
import json
from pathlib import Path
from radio_gs.querying.reference_response import main as fit

parser = argparse.ArgumentParser()
parser.add_argument('--inputs', required=True)
parser.add_argument('--query', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
inputs = json.loads(Path(args.inputs).read_text())
fit(['--scene', inputs['scene'], '--query', args.query, '--evidence', inputs['detections'], '--output', args.output])
