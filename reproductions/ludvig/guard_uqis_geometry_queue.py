#!/usr/bin/env python3
"""Thermally guard one exact UQIS geometry queue on a selected physical GPU."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def _temperature(gpu: int) -> int:
    value = subprocess.run(
        [
            "nvidia-smi", "-i", str(gpu),
            "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=3,
    ).stdout.strip()
    return int(value)


def _matching_trainers(output_root: Path) -> list[int]:
    marker = str(output_root.resolve())
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace")
        if "gaussian-splatting-ludvig-audit/train.py" in command and marker in command:
            result.append(int(entry.name))
    return sorted(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-pid", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--pause-c", type=int, default=88)
    parser.add_argument("--resume-c", type=int, default=65)
    parser.add_argument("--poll-seconds", type=float, default=3.0)
    parser.add_argument("--event-log", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.resume_c < args.pause_c < 100:
        raise ValueError("thermal thresholds must satisfy 0 < resume < pause < 100")
    args.event_log.parent.mkdir(parents=True, exist_ok=True)
    paused: set[int] = set()
    events = []
    try:
        while Path(f"/proc/{args.queue_pid}").exists():
            temperature = _temperature(args.gpu)
            trainers = _matching_trainers(args.output_root)
            for pid in trainers:
                if temperature >= args.pause_c and pid not in paused:
                    os.kill(pid, signal.SIGSTOP)
                    paused.add(pid)
                    events.append(
                        {
                            "time": datetime.now(timezone.utc).isoformat(),
                            "event": "thermal_pause",
                            "pid": pid,
                            "temperature_c": temperature,
                        }
                    )
                elif temperature <= args.resume_c and pid in paused:
                    os.kill(pid, signal.SIGCONT)
                    paused.remove(pid)
                    events.append(
                        {
                            "time": datetime.now(timezone.utc).isoformat(),
                            "event": "thermal_resume",
                            "pid": pid,
                            "temperature_c": temperature,
                        }
                    )
            paused.intersection_update(trainers)
            payload = {
                "status": "guarding",
                "queue_pid": args.queue_pid,
                "gpu": args.gpu,
                "pause_c": args.pause_c,
                "resume_c": args.resume_c,
                "last_temperature_c": temperature,
                "active_trainers": trainers,
                "paused_trainers": sorted(paused),
                "events": events,
            }
            args.event_log.write_text(json.dumps(payload, indent=2) + "\n")
            time.sleep(args.poll_seconds)
    finally:
        for pid in tuple(paused):
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
        payload = {
            "status": "complete",
            "queue_pid": args.queue_pid,
            "gpu": args.gpu,
            "pause_c": args.pause_c,
            "resume_c": args.resume_c,
            "events": events,
        }
        args.event_log.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
