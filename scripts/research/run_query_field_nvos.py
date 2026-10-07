"""Freeze all prompt arms, source-denied replay, then score once."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import torch

from radio_gs.query_field.storage import sha256


def run_prediction(config_path, job, output, log, deny=()):
    command = [sys.executable, "-m", "radio_gs.query_field.public_nvos", "predict",
               "--config", str(config_path), "--job", job, "--output", str(output)]
    for root in deny:
        command += ["--deny-root", str(root)]
    with Path(log).open("w") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)


def replay(config_path, predictions, root):
    """Replay a completed fixed cohort; reusable across source-training studies."""
    config_path, predictions, root = map(Path, [config_path, predictions, root])
    if (root/"cold_replay.json").exists():
        raise FileExistsError("preserve completed cold replay")
    config = json.loads(config_path.read_text())
    for job in config["jobs"]:
        record = json.loads((predictions/job["name"]/"prediction.json").read_text())
        if record['configuration'] != config:
            raise ValueError("cold replay configuration differs from prediction")
        if sha256(predictions/job["name"]/"predictions.pt") != record['predictions_sha256']:
            raise ValueError("cold replay prediction payload changed")
    temporary = Path(tempfile.mkdtemp(prefix="query_field_nvos_cold_"))
    copied = {}
    cold = json.loads(json.dumps(config))
    for job in cold["jobs"]:
        for key in ["field", "camera_fixture", "positive", "negative"]:
            source = Path(job[key]).resolve(strict=True)
            if str(source) not in copied:
                destination = temporary / f"input_{len(copied)}"
                if source.is_dir():
                    shutil.copytree(source, destination)
                else:
                    shutil.copyfile(source, destination)
                copied[str(source)] = str(destination)
            job[key] = copied[str(source)]
    cold_config = temporary / "config.json"
    cold_config.write_text(json.dumps(cold, indent=2)+"\n")
    deny = [Path("output").resolve(),
            Path("/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/NVOS")]
    # Verify that the denial hook itself rejects an original scene-file open.
    source_fixture = config["jobs"][0]["camera_fixture"]
    guard_code = "from radio_gs.query_field.public_nvos import deny_original_inputs; import sys; deny_original_inputs(sys.argv[2:]); open(sys.argv[1], 'rb')"
    guard = subprocess.run([sys.executable, "-c", guard_code, source_fixture, *map(str, deny)],
                           capture_output=True, text=True)
    if guard.returncode == 0 or "source input denied" not in guard.stderr:
        raise RuntimeError("cold replay guard did not reject original scene input")
    comparisons = []
    for job in cold["jobs"]:
        run_prediction(cold_config, job["name"], temporary/"predictions",
                       root/f"cold_{job['name']}.log", deny)
        reference = torch.load(predictions/job["name"]/"predictions.pt", map_location="cpu", weights_only=True)
        replay = torch.load(temporary/"predictions"/job["name"]/"predictions.pt", map_location="cpu", weights_only=True)
        equal = {key: torch.equal(reference[key], replay[key]) for key in reference}
        if not all(equal.values()):
            raise RuntimeError(f"cold prompt replay differs: {job['name']}: {equal}")
        comparisons.append(dict(job=job["name"], exact=equal))
        print("cold replay", job["name"], equal, flush=True)
    report = dict(temporary=str(temporary), denied_roots=list(map(str, deny)),
                  original_open_rejected=True, comparisons=comparisons,
                  isolation="fresh process plus Python open audit; not OS sandbox",
                  configuration_sha256=sha256(config_path))
    (root/"cold_replay.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    root = Path("output/query_field_20261007/nvos_fern_v1")
    config_path = Path("configs/query_field/nvos_fern_prediction.json")
    config = json.loads(config_path.read_text())
    predictions = root / "public"
    predictions.mkdir(exist_ok=False)
    for job in config["jobs"]:
        run_prediction(config_path, job["name"], predictions, root/f"predict_{job['name']}.log")
        print("predicted", job["name"], flush=True)
    replay(config_path, predictions, root)
    command = [sys.executable, "-m", "radio_gs.query_field.public_nvos", "score",
               "--config", "configs/query_field/nvos_fern_scoring.json", "--output", str(predictions)]
    with (root/"score.log").open("w") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
    print((predictions/"evaluation.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
