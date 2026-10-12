"""Cold fixed-first public edit-penalized query without historical scene IO."""
import torch
from scripts.research.run_query_field_reference_edit import ROOT,design
from scripts.research.probe_query_field_reference_extent import main


if __name__=='__main__':
    torch.set_num_threads(4)
    main(design_loader=design,root=ROOT,arm='edit',observation_weight=.5,regularizer='edit',
        report_path='docs/experiments/2026-10-12-reference-edit-cold-probe.json')
