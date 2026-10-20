from pathlib import Path
from types import SimpleNamespace
import tempfile
import numpy as np
import pytest
import torch
from scripts.research.tmm_ram_predictions import RamPredictionBank,public_query_files


def test_ram_predictions_survive_reader_recreation_and_detect_changes():
    bank=RamPredictionBank();run=SimpleNamespace(root=Path('/a/run'))
    try:
        expected=torch.tensor([[.1,.2],[.3,.4]])
        bank.add(run,'arm',expected)
        recovered=RamPredictionBank(bank.root)
        torch.testing.assert_close(recovered.read(run,'arm',2),expected)
        path=bank.root/next(iter(bank.entries.values()))['file']
        array=np.load(path,mmap_mode='r+');array[0,0]=.9;array.flush();del array
        with pytest.raises(ValueError,match='changed'):recovered.read(run,'arm',2)
    finally:bank.close()


def test_ram_bank_rejects_over_budget_write():
    bank=RamPredictionBank(limit=5000)
    try:
        with pytest.raises(ValueError,match='budget'):
            bank.add(SimpleNamespace(root=Path('/a/run')),'arm',torch.zeros(2000,1))
    finally:bank.close()


def test_query_directory_hashes_actual_public_files(tmp_path):
    (tmp_path/'queries').mkdir()
    for m in ['text','image','point_2d','point_3d']:(tmp_path/'queries'/f'{m}.json').write_text('{}')
    files=public_query_files(tmp_path)
    assert len(files)==4 and all(p.is_file() for p in files)
    assert public_query_files(files[0])==[files[0]]


def test_ram_bank_rejects_persistent_storage(tmp_path):
    with pytest.raises(ValueError,match='/dev/shm'):RamPredictionBank(tmp_path)


def test_partial_job_cleanup_preserves_other_completed_job():
    bank=RamPredictionBank();done=SimpleNamespace(root=Path('/done'));partial=SimpleNamespace(root=Path('/partial'))
    try:
        value=torch.tensor([[.4]])
        bank.add(done,'arm',value);bank.add(partial,'arm',value)
        bank.discard_unsealed(partial)
        assert bank.bytes==4
        torch.testing.assert_close(bank.read(done,'arm',1),value)
        bank.add(partial,'arm',value)
    finally:bank.close()
