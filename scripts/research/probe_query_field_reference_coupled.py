"""Fixed-first-case cold public coupled query, before consulting its answer."""
import torch
from scripts.research.run_query_field_reference_coupled import ROOT,design
from scripts.research.probe_query_field_reference_extent import main


if __name__=='__main__':
    torch.set_num_threads(4)
    main(design_loader=design,root=ROOT,arm='coupled',observation_weight=.5,
         report_path='docs/experiments/2026-10-12-reference-coupled-cold-probe.json')
