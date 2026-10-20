import pytest
import torch
from scripts.research.tmm_transient_runtime import run_with_gpu_retry


def fake_cuda(monkeypatch,free):
    selected=[]
    monkeypatch.setattr(torch.cuda,'device_count',lambda:len(free))
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda i:(free[i],100))
    monkeypatch.setattr(torch.cuda,'set_device',selected.append)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    return selected


def test_oom_retries_current_job_and_preserves_completed_predictions(monkeypatch):
    free=[80,20];selected=fake_cuda(monkeypatch,free);events=[]
    completed={'already_sealed':object()};original=completed['already_sealed'];attempts=[]
    def operation():
        attempts.append(1)
        if len(attempts)==1:
            free[:]=[10,90]
            raise torch.cuda.OutOfMemoryError('another process consumed device0')
        assert completed['already_sealed'] is original
        return 'new job'
    result=run_with_gpu_retry(operation,event=events.append,minimum_free_bytes=50,sleep=lambda _:None)
    assert result=='new job' and selected==[0,1]
    assert [e['state'] for e in events]==['running','retry_after_cuda_oom','running']


def test_waits_without_running_on_insufficient_device(monkeypatch):
    free=[10];selected=fake_cuda(monkeypatch,free);events=[]
    def sleep(_):free[0]=70
    result=run_with_gpu_retry(lambda:42,event=events.append,minimum_free_bytes=50,sleep=sleep)
    assert result==42 and selected==[0]
    assert events[0]['state']=='waiting_for_gpu'


def test_logic_errors_are_not_retried_as_resource_failures(monkeypatch):
    fake_cuda(monkeypatch,[80])
    def operation():raise ValueError('input contract changed')
    with pytest.raises(ValueError,match='input contract'):
        run_with_gpu_retry(operation,event=lambda _:None,minimum_free_bytes=50)
