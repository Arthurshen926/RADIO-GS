import torch
import numpy as np
import pytest
from radio_gs.querying.semantic_point_readout import classify_elements
from radio_gs.scannet_constants import compute_split_metrics

class Field:
 observed=torch.tensor([True,False,True])
 def logits(self,q,m):return torch.tensor([[0.,1.],[1.,0.],[2.,0.]])

def test_raw_class_ids_and_unobserved_misses_survive_categorical_readout():
 r=classify_elements(Field(),torch.zeros(2,4),[5,33]);assert r['raw_class_ids'].tolist()==[33,0,5]
 m=compute_split_metrics(r['raw_class_ids'].numpy(),np.array([33,5,5]),[5,33]);assert m['per_class']['5']['iou']==.5

def test_no_duplicate_or_ignored_class_queries():
 for ids in [[1,1],[0,1]]:
  with pytest.raises(ValueError):classify_elements(Field(),torch.zeros(2,4),ids)
