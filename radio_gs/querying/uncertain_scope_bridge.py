"""Connect candidate regions across a single uncertain node, without selecting it."""
import torch


def candidate_bridge_edges(candidate, unary, neighbors, spatial_edges, *, enabled=False):
    direct=spatial_edges&candidate[:,None]&candidate[neighbors]
    if not enabled:return direct,torch.zeros_like(candidate)
    adjacent=(spatial_edges&candidate[neighbors]).sum(1)
    bridge=(~candidate)&(unary>=.25)&(adjacent>=2)
    # No bridge-to-bridge edge: a chain of uncertain nodes is not evidence.
    edges=spatial_edges&((candidate[:,None]&(candidate[neighbors]|bridge[neighbors]))|(bridge[:,None]&candidate[neighbors]))
    return edges,bridge
