"""Cold fixed-first public grouped query without historical scene IO."""
import torch
from scripts.research.run_query_field_reference_groups import ROOT,design
from scripts.research.probe_query_field_reference_extent import main


if __name__=='__main__':
    torch.set_num_threads(4)
    main(design_loader=design,root=ROOT,arm='parts',observation_weight=.5,grouping='incidence',
        report_path='docs/experiments/2026-10-12-reference-groups-cold-probe.json')
