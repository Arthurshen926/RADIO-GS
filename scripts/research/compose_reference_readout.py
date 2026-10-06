"""Bind separately trained response and selected-set outputs to one query/map.

The output contract fixes which head is scored; no metric or annotation chooses
between predictions. Both heads consume the same reference-identified evidence.
"""
import argparse
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--response-field',required=True);p.add_argument('--selection-field',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    response=torch.load(a.response_field,map_location='cpu',weights_only=False);selection=torch.load(a.selection_field,map_location='cpu',weights_only=False)
    result=dict(texts=selection['texts'],scores=response['elements'],semantic_geometry=response['geometry'],semantic_observed=response['observed'],selection_scores=selection['selection_scores'],selection_geometry=selection['selection_geometry'],selection_observed=selection['selection_observed'],readout_contract='semantic_probability_and_independent_selected_set',query_intent='reference_instance',response_field=a.response_field,selection_field=a.selection_field)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);torch.save(result,a.output)


if __name__=='__main__':main()
