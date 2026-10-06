"""Execution-only memory control: retain exact saved tensors on host memory.

Runs an unchanged source stage under PyTorch's saved-tensor CPU offload. This
does not alter batches, teacher losses, precision, optimizer or field density.
The caller must record this wrapper in its actual-command provenance.
"""
import runpy
import sys
import torch

if len(sys.argv) < 2:
    raise ValueError('Source guard/entrypoint required')
entry = sys.argv[1]
sys.argv = sys.argv[1:]
with torch.autograd.graph.save_on_cpu(pin_memory=True):
    runpy.run_path(entry, run_name='__main__')
