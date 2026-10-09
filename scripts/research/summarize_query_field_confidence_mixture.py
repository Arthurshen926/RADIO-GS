"""Complete confidence repair with the same full-cohort report and audits."""
import argparse
from pathlib import Path

import torch

from radio_gs.query_field.confidence_mixture_evidence import ConfidenceMixtureEvidence
from scripts.research import summarize_query_field_mixture_functions as summary
from scripts.research import diagnose_query_field_mixture_activation as points
from scripts.research import diagnose_query_field_mixture_signed as signed
from scripts.research import audit_query_field_reference_budgets as budgets
from scripts.research import plot_query_field_mixture_functions as plot
from scripts.research.run_query_field_confidence_mixture_functions import ROOT, ARMS, frozen_design


def configure():
    for module in [summary, points, signed, budgets, plot]:
        module.ROOT = ROOT; module.ARMS = ARMS
    summary.frozen_design = frozen_design
    summary.RESULT = Path('docs/experiments/2026-10-09-confidence-mixture-results.json')
    points.RELATION_CLASS = ConfidenceMixtureEvidence


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['diagnose', 'summarize', 'plot', 'budgets'])
    args = p.parse_args(); torch.set_num_threads(4); configure()
    {'diagnose': summary.diagnose, 'summarize': summary.main, 'plot': plot.main, 'budgets': budgets.main}[args.stage]()
