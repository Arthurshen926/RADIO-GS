"""Keep sealed CPU predictions alive across recoverable shared-GPU failures."""
import gc
import time

import torch


def available_device(minimum_free_bytes):
    devices=[(torch.cuda.mem_get_info(i)[0],i) for i in range(torch.cuda.device_count())]
    free,index=max(devices,default=(0,None))
    return index if free>=minimum_free_bytes else None


def run_with_gpu_retry(operation, *, event, minimum_free_bytes=16*2**30, sleep=time.sleep):
    """Retry only this operation; the caller retains completed CPU predictions.

    No dataset rule, parameter or prediction from a completed job is changed.
    Other processes are never stopped. Waiting/retries are recorded by caller.
    This protects against CUDA OOM, not process termination or host failure.
    """
    attempt=0
    while True:
        index=available_device(minimum_free_bytes)
        if index is None:
            event(dict(state='waiting_for_gpu',attempt=attempt))
            sleep(15)
            continue
        torch.cuda.set_device(index)
        attempt+=1
        event(dict(state='running',attempt=attempt,cuda_device=index))
        failure=None
        try:
            return operation()
        except torch.cuda.OutOfMemoryError as error:
            failure=str(error)
            # Release tensors held by failed stack frames before retrying.
            error.__traceback__=None
        gc.collect()
        torch.cuda.empty_cache()
        event(dict(state='retry_after_cuda_oom',attempt=attempt,cuda_device=index,error=failure))
        sleep(5)
