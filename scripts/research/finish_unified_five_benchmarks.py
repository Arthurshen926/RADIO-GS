"""Finish reports for an already running local cohort; never resubmit failed work."""
import argparse
import os
from pathlib import Path
import time

from radio_gs.query_field.storage import sha256
from radio_gs.query_field.unified_five_benchmark import control, read
from scripts.research.summarize_unified_five_benchmarks import summarize


def finish(config, supervisor_pid):
    store=control(config)
    summary_source=Path(__file__).with_name("summarize_unified_five_benchmarks.py")
    frozen=sha256(summary_source)
    while not (store.root / "scoring_complete.json").exists():
        try:os.kill(supervisor_pid,0)
        except ProcessLookupError:
            if not (store.root / "scoring_complete.json").exists():
                store.json("closure_failed.json",dict(supervisor_pid=supervisor_pid,
                    reason="cohort supervisor exited without complete all-five scoring",
                    progress=read(store.root / "progress.json")))
                raise RuntimeError("full-cohort closure incomplete; failures remain visible")
        time.sleep(15)
    if sha256(summary_source)!=frozen:raise ValueError("summary implementation changed")
    report=summarize(config)
    store.json("closure_complete.json",dict(five_benchmarks_complete=True,
        summary_sha256=sha256(store.root / "summary.json"),summary_source_sha256=frozen,
        architecture=report["architecture"],accuracy_promotion=False))
    print("Five-task summary complete:",store.root / "summary.json",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/query_field/unified_five_benchmark.json")
    p.add_argument("--supervisor-pid",type=int,required=True)
    a=p.parse_args();finish(read(a.config),a.supervisor_pid)
