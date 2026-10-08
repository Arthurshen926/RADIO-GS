"""Complete soft-target extension with matched-source functional diagnostics."""
import argparse
from pathlib import Path

import torch

from scripts.research import summarize_query_field_paired_functions as common
from scripts.research import diagnose_query_field_paired_activation as activation
from scripts.research import diagnose_query_field_signed_activation as signed
from scripts.research.run_query_field_soft_functions import ROOT, ARMS, frozen_design


def main(diagnose=False):
    common.ROOT = ROOT; common.ARMS = ARMS; common.frozen_design = frozen_design
    common.RESULT_PATH = Path('docs/experiments/2026-10-09-soft-paired-function-results.json')
    if diagnose:
        common.diagnose(); activation.main(); signed.main()
    else:
        common.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--diagnose', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(4); main(args.diagnose)
