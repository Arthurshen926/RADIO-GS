#!/usr/bin/env python3
"""Keep selected visible CUDA devices warm until stopped.

This utility is intentionally operational rather than experimental. It should be
started only when the machine owner wants idle GPUs reserved between main runs.
Stop it before launching real experiments on the same devices.
"""

from __future__ import annotations

import argparse
import os
import signal
import threading
import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DevicePlan:
    visible_index: int
    target_bytes: int
    matrix_size: int


def _parse_gpus(raw: str) -> list[int]:
    gpus = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not gpus:
        raise ValueError("at least one visible GPU index is required")
    return gpus


def _gib(num_bytes: int) -> float:
    return num_bytes / float(1024**3)


def _reserve_memory(target_bytes: int, *, device: torch.device, chunk_mib: int) -> list[torch.Tensor]:
    chunks: list[torch.Tensor] = []
    chunk_elems = max(1, chunk_mib * 1024 * 1024 // 2)
    reserved = 0
    while reserved < target_bytes:
        remaining = target_bytes - reserved
        elems = min(chunk_elems, max(1, remaining // 2))
        try:
            chunks.append(torch.empty((elems,), dtype=torch.float16, device=device))
            reserved += elems * 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
    return chunks


def _worker(
    plan: DevicePlan,
    *,
    chunk_mib: int,
    heartbeat_sec: float,
    safety_free_mib: int,
    sync_every: int,
    sleep_ms: float,
    stop_event: threading.Event,
) -> None:
    torch.cuda.set_device(plan.visible_index)
    device = torch.device(f"cuda:{plan.visible_index}")

    n = plan.matrix_size
    a = torch.randn((n, n), dtype=torch.float16, device=device)
    b = torch.randn((n, n), dtype=torch.float16, device=device)
    c = torch.empty((n, n), dtype=torch.float16, device=device)

    initial_free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free_floor = max(
        int(total_bytes * (1.0 - min(plan.target_bytes / max(total_bytes, 1), 0.98))),
        safety_free_mib * 1024 * 1024,
    )
    target_bytes = max(0, initial_free_bytes - free_floor)
    reserve = _reserve_memory(target_bytes, device=device, chunk_mib=chunk_mib)

    last_heartbeat = 0.0
    iterations = 0
    while not stop_event.is_set():
        c = torch.mm(a, b, out=c)
        a, c = c, a
        iterations += 1
        if iterations % sync_every == 0:
            torch.cuda.synchronize(device)
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)
        now = time.time()
        if now - last_heartbeat >= heartbeat_sec:
            free_now, total_now = torch.cuda.mem_get_info(device)
            print(
                (
                    f"[gpu-placeholder] pid={os.getpid()} visible_cuda={plan.visible_index} "
                    f"reserved_chunks={len(reserve)} free={_gib(free_now):.2f}GiB/"
                    f"{_gib(total_now):.2f}GiB matrix={n} iter={iterations}"
                ),
                flush=True,
            )
            last_heartbeat = now

    torch.cuda.synchronize(device)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1", help="Visible CUDA device indices, e.g. 0,1")
    parser.add_argument(
        "--memory_fraction",
        type=float,
        default=0.80,
        help="Fraction of currently free memory to reserve on each visible device",
    )
    parser.add_argument("--matrix_size", type=int, default=8192)
    parser.add_argument("--chunk_mib", type=int, default=512)
    parser.add_argument("--heartbeat_sec", type=float, default=60.0)
    parser.add_argument(
        "--sync_every",
        type=int,
        default=16,
        help="Synchronize and optionally sleep after this many GEMM iterations",
    )
    parser.add_argument(
        "--sleep_ms",
        type=float,
        default=0.0,
        help="Milliseconds to sleep after each synchronization window",
    )
    parser.add_argument(
        "--safety_free_mib",
        type=int,
        default=3072,
        help="Keep at least this much free memory per visible GPU after reservation",
    )
    parser.add_argument(
        "--duration_sec",
        type=float,
        default=0.0,
        help="Optional finite duration for smoke tests; 0 means run until signalled",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not 0.0 < args.memory_fraction < 1.0:
        raise ValueError("--memory_fraction must be between 0 and 1")
    if args.sync_every <= 0:
        raise ValueError("--sync_every must be positive")
    if args.sleep_ms < 0:
        raise ValueError("--sleep_ms must be non-negative")

    stop_event = threading.Event()

    def _handle_signal(signum: int, _frame: object) -> None:
        print(f"[gpu-placeholder] received signal {signum}; stopping", flush=True)
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    plans: list[DevicePlan] = []
    for visible_index in _parse_gpus(args.gpus):
        torch.cuda.set_device(visible_index)
        free_bytes, _total_bytes = torch.cuda.mem_get_info(visible_index)
        plans.append(
            DevicePlan(
                visible_index=visible_index,
                target_bytes=int(free_bytes * args.memory_fraction),
                matrix_size=args.matrix_size,
            )
        )

    print(
        (
            f"[gpu-placeholder] pid={os.getpid()} visible_devices={args.gpus} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
        ),
        flush=True,
    )

    threads = [
        threading.Thread(
            target=_worker,
            kwargs={
                "plan": plan,
                "chunk_mib": args.chunk_mib,
                "heartbeat_sec": args.heartbeat_sec,
                "safety_free_mib": args.safety_free_mib,
                "sync_every": args.sync_every,
                "sleep_ms": args.sleep_ms,
                "stop_event": stop_event,
            },
            daemon=False,
        )
        for plan in plans
    ]
    for thread in threads:
        thread.start()

    start_time = time.time()
    try:
        while any(thread.is_alive() for thread in threads):
            if args.duration_sec > 0 and time.time() - start_time >= args.duration_sec:
                stop_event.set()
            time.sleep(0.5)
    finally:
        stop_event.set()
        for thread in threads:
            thread.join()


if __name__ == "__main__":
    main()
