"""Check first-query IO in a fresh process, without prior overload compilation."""
import subprocess
import sys
import textwrap


def test_cold_warmup_matches_actual_solve_under_file_denial():
    code=textwrap.dedent('''
        import numpy as np
        import torch
        from unittest.mock import patch
        from radio_gs.query_field.extent_factors import ExtentFactors
        from radio_gs.query_field.extent_decision_linked import LinkedExtentDecision
        from radio_gs.query_field.extent_runtime import warm_extent_kernel
        state=dict(pointer=torch.tensor([0,1]),index=torch.tensor([0]),alpha=torch.tensor([.8]),
            visibility=torch.tensor([.8]),atom=torch.tensor([0]))
        engine=LinkedExtentDecision(ExtentFactors({'views.0.'+k:v for k,v in state.items()},1),1)
        warm_extent_kernel(engine)
        with patch('builtins.open',side_effect=AssertionError('query IO')),patch('io.open',side_effect=AssertionError('query IO')):
            selected,report=engine.solve(np.array([False]),np.array([1.]),np.array([True]),np.array([False]),np.ones(1),np.zeros(1))
        assert selected.tolist()==[True] and not report['fallback']
    ''')
    subprocess.run([sys.executable,'-c',code],check=True)
