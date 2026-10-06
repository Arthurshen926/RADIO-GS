"""Source-learned category competition correction in fixed native coordinates."""
import torch
from torch.nn import functional as F


def correct_category_logits(logits, query, state):
    direction = state['direction'].to(query.device).float()
    correction = (F.normalize(query.float(), dim=-1) @ direction).tanh()
    return logits + correction[None]
