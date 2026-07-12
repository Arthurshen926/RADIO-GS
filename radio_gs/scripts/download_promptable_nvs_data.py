"""Download the public NVOS/SPIn-NeRF benchmark assets used by RADIO-GS.

The benchmark is split across three official public shares:

* NVOS annotations (Dropbox),
* the undistorted NeX/LLFF captures referenced by NVOS (SharePoint), and
* SPIn-NeRF data and multiview annotations (Google Drive).

This helper deliberately downloads immutable archives/trees without changing
their contents.  It writes ``download_manifest.json`` records next to each
download so that an evaluation run can retain source URLs, byte sizes and
SHA-256 digests.  It does not download or execute any baseline implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import http.cookiejar
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional


NVOS_ANNOTATIONS_URL = (
    "https://www.dropbox.com/sh/sdgr4mewkhjsg00/"
    "AACIKecIwzCHCGma5kkKyLTpa?dl=1"
)
NVOS_LLFF_SHARE_URL = (
    "https://vistec-my.sharepoint.com/:f:/g/personal/"
    "pakkapon_p_s19_vistec_ac_th/"
    "ErjPRRL9JnFIp8MN6d1jEuoB3XVoxJkffPjfoPyhHkj0dg?e=qIunN0"
)
NVOS_LLFF_SHARE_PATH = "public/VLL/NeX/modified_dataset/frontface"
SPIN_DRIVE_URL = (
    "https://drive.google.com/drive/folders/"
    "1N7D4-6IutYD40v9lfXGSVbWrd47UdJEC?usp=share_link"
)
SPIN_LLFF_DRIVE_URL = (
    "https://drive.google.com/drive/folders/"
    "128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1"
)
SPIN_LLFF_MIRROR_URL = (
    "https://huggingface.co/datasets/YouLiXiya/nerf/resolve/"
    "8bb2c1f48e1ee51a94bce0c1634dd04795f92faa/nerf_llff_data.zip?download=true"
)
SPIN_LLFF_MIRROR_SIZE = 1_780_545_599
SPIN_LLFF_MIRROR_SHA256 = (
    "b8be42c77ce345e647812cb69d1f92d2a85159f2464847e99458e53d13cb1d96"
)
SPIN_FORK_FILE_URL = (
    "https://drive.google.com/file/d/"
    "16_y_Nnh19Qhml0bg9RYR-hav0YOpWKuw/view"
)
SPIN_LEGO_FILE_URL = (
    "https://drive.google.com/file/d/"
    "1PG-KllCv4vSRPO7n5lpBjyTjlUyT8Nag/view"
)
SPIN_LEGO_MIRROR_URL = (
    "https://models.nmb.ai/ARF/lego_real_night_radial.tar.gz"
)
SPIN_LEGO_MIRROR_SIZE = 586_506_544
SPIN_LEGO_MIRROR_SHA256 = (
    "fe97e2698d88525f9937f37bbd05ad01c277f602ed932e2ccee778ff4519e06b"
)
SPIN_FORK_EXPECTED_SIZE = 481_028_506
SPIN_PINECONE_MIRROR_URL = (
    "https://huggingface.co/datasets/YouLiXiya/nerf/resolve/"
    "a92cf26ff593418fb5d46db92e462ac55ab17d0e/nerf_real_360.zip?download=true"
)
SPIN_PINECONE_MIRROR_SHA256 = (
    "e5996aa08cf9a22c28adc21d9321ca302bd737ad71d9959dcdab825d6981b0ad"
)
SPIN_TANDT_3DGS_URL = (
    "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/"
    "datasets/input/tandt_db.zip"
)


def _sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_request(
    opener: urllib.request.OpenerDirector,
    url: str,
    *,
    attempts: int = 20,
) -> Dict[str, Any]:
    last_error: Optional[BaseException] = None
    for attempt in range(attempts):
        try:
            with opener.open(url, timeout=90) as response:
                return json.load(response)
        except (
            OSError,
            urllib.error.URLError,
            http.client.IncompleteRead,
            json.JSONDecodeError,
        ) as error:
            last_error = error
            if attempt + 1 == attempts:
                break
            time.sleep(min(2**attempt, 20))
    assert last_error is not None
    raise last_error


def _copy_response(
    opener: urllib.request.OpenerDirector,
    url: str,
    destination: Path,
    *,
    expected_size: Optional[int] = None,
    chunk_size: int = 8 << 20,
    attempts: int = 12,
) -> Dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    last_error: Optional[BaseException] = None
    for attempt in range(attempts):
        digest = hashlib.sha256()
        byte_count = 0
        try:
            with opener.open(url, timeout=180) as response, partial.open("wb") as output:
                while chunk := response.read(chunk_size):
                    output.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
            break
        except (OSError, urllib.error.URLError, http.client.IncompleteRead) as error:
            last_error = error
            if attempt + 1 == attempts:
                raise
            time.sleep(min(2**attempt, 20))
    else:  # pragma: no cover - the retry loop either breaks or raises
        assert last_error is not None
        raise last_error
    if expected_size is not None and byte_count != int(expected_size):
        partial.unlink(missing_ok=True)
        raise IOError(
            f"Size mismatch for {destination}: got {byte_count}, expected {expected_size}"
        )
    partial.replace(destination)
    return {
        "path": str(destination),
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
    }


def _write_manifest(root: Path, payload: Mapping[str, Any]) -> Path:
    path = root / "download_manifest.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def download_nvos_annotations(root: Path, *, force: bool = False) -> Path:
    target_root = root / "NVOS"
    archive = target_root / "nvos-data.zip"
    target_root.mkdir(parents=True, exist_ok=True)
    if force or not archive.is_file():
        opener = urllib.request.build_opener()
        record = _copy_response(opener, NVOS_ANNOTATIONS_URL, archive)
    else:
        record = {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": _sha256(archive),
        }
    record["url"] = NVOS_ANNOTATIONS_URL
    record["role"] = "official_nvos_annotations"
    manifest = {
        "dataset": "NVOS",
        "downloaded_at_unix": int(time.time()),
        "files": [record],
    }
    _write_manifest(target_root, manifest)
    return archive


def _sharepoint_context(
    share_url: str,
) -> tuple[urllib.request.OpenerDirector, str, str]:
    cookies = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookies))
    with opener.open(share_url, timeout=90) as response:
        html = response.read().decode("utf-8", errors="replace")
    match = re.search(r"var _spPageContextInfo=(\{.*?\});", html)
    if match is None:
        raise RuntimeError("SharePoint page did not expose _spPageContextInfo")
    context = json.loads(match.group(1))
    drive_info = context.get("driveInfo", {})
    drive_url = drive_info.get(".driveUrl")
    access_token = drive_info.get(".driveAccessToken")
    if not drive_url or not access_token:
        raise RuntimeError("SharePoint page did not expose anonymous drive access")
    return opener, str(drive_url), str(access_token)


def _with_sharepoint_token(url: str, token_query: str) -> str:
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{token_query}"


def _sharepoint_children(
    opener: urllib.request.OpenerDirector,
    drive_url: str,
    token_query: str,
    *,
    root_path: Optional[str] = None,
    item_id: Optional[str] = None,
) -> Iterator[Mapping[str, Any]]:
    if (root_path is None) == (item_id is None):
        raise ValueError("Specify exactly one of root_path or item_id")
    if root_path is not None:
        quoted = urllib.parse.quote(root_path.strip("/"), safe="/")
        endpoint = f"{drive_url}/root:/{quoted}:/children"
    else:
        endpoint = f"{drive_url}/items/{urllib.parse.quote(str(item_id), safe='')}/children"
    next_url: Optional[str] = _with_sharepoint_token(endpoint, token_query)
    while next_url:
        payload = _json_request(opener, next_url)
        yield from payload.get("value", [])
        next_url = payload.get("@odata.nextLink")
        if next_url and "access_token=" not in next_url:
            next_url = _with_sharepoint_token(next_url, token_query)


def download_nvos_llff(root: Path, *, force: bool = False) -> Path:
    """Recursively fetch the exact undistorted LLFF tree linked by NVOS."""
    target_root = root / "NVOS" / "llff_undistorted"
    target_root.mkdir(parents=True, exist_ok=True)
    opener, drive_url, token_query = _sharepoint_context(NVOS_LLFF_SHARE_URL)
    records: List[Dict[str, Any]] = []

    def visit(items: Iterable[Mapping[str, Any]], relative_root: Path) -> None:
        for item in sorted(items, key=lambda value: str(value.get("name", ""))):
            name = str(item.get("name", ""))
            if not name or name in {".", ".."} or Path(name).name != name:
                raise ValueError(f"Unsafe SharePoint item name: {name!r}")
            relative = relative_root / name
            destination = target_root / relative
            if "folder" in item:
                destination.mkdir(parents=True, exist_ok=True)
                children = _sharepoint_children(
                    opener,
                    drive_url,
                    token_query,
                    item_id=str(item["id"]),
                )
                visit(children, relative)
                continue
            download_url = item.get("@content.downloadUrlNoAuth") or item.get(
                "@content.downloadUrl"
            )
            if not download_url:
                raise RuntimeError(f"No download URL for SharePoint file {relative}")
            expected_size = int(item.get("size", 0))
            if not force and destination.is_file() and destination.stat().st_size == expected_size:
                record = {
                    "path": str(relative),
                    "bytes": expected_size,
                    "sha256": _sha256(destination),
                    "status": "existing",
                }
            else:
                record = _copy_response(
                    opener,
                    str(download_url),
                    destination,
                    expected_size=expected_size,
                )
                record["path"] = str(relative)
                record["status"] = "downloaded"
            records.append(record)
            print(f"[{len(records):04d}] {relative} ({expected_size} bytes)", flush=True)

    top_level = _sharepoint_children(
        opener,
        drive_url,
        token_query,
        root_path=NVOS_LLFF_SHARE_PATH,
    )
    visit(top_level, Path())
    _write_manifest(
        target_root,
        {
            "dataset": "NVOS",
            "role": "official_undistorted_llff_images",
            "source_share_url": NVOS_LLFF_SHARE_URL,
            "source_tree_path": NVOS_LLFF_SHARE_PATH,
            "downloaded_at_unix": int(time.time()),
            "files": records,
            "total_bytes": sum(int(record["bytes"]) for record in records),
        },
    )
    return target_root


def download_spin_official(root: Path, *, force: bool = False) -> Path:
    """Fetch the official SPIn-NeRF Google Drive folder with ``gdown``."""
    target = root / "SPIn-NeRF" / "official"
    target.mkdir(parents=True, exist_ok=True)
    gdown = shutil.which("gdown")
    if gdown is None:
        raise RuntimeError("gdown is required for the public SPIn-NeRF Drive folder")
    command = [
        gdown,
        "--folder",
        "--remaining-ok",
        "--continue",
        SPIN_DRIVE_URL,
        "-O",
        str(target) + os.sep,
    ]
    if force:
        command.remove("--continue")
    subprocess.run(command, check=True)
    records = []
    for path in sorted(target.rglob("*")):
        if (
            not path.is_file()
            or path.name.endswith(".part")
            or path.name == "download_manifest.json"
        ):
            continue
        records.append(
            {
                "path": str(path.relative_to(target)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    _write_manifest(
        target,
        {
            "dataset": "SPIn-NeRF",
            "role": "official_drive_folder",
            "source_url": SPIN_DRIVE_URL,
            "downloaded_at_unix": int(time.time()),
            "files": records,
            "total_bytes": sum(int(record["bytes"]) for record in records),
        },
    )
    return target


def _run_gdown(
    url: str,
    target: Path,
    *,
    folder: bool,
    force: bool,
) -> None:
    gdown = shutil.which("gdown")
    if gdown is None:
        raise RuntimeError("gdown is required for public Google Drive assets")
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [gdown]
    if folder:
        target.mkdir(parents=True, exist_ok=True)
        command.extend(["--folder", "--remaining-ok"])
    if not force:
        command.append("--continue")
    command.extend([url, "-O", str(target) + (os.sep if folder else "")])
    subprocess.run(command, check=True)


def _record_existing_or_download(
    opener: urllib.request.OpenerDirector,
    *,
    url: str,
    destination: Path,
    expected_size: int,
    expected_sha256: Optional[str] = None,
    force: bool,
) -> Dict[str, Any]:
    if not force and destination.is_file() and destination.stat().st_size == expected_size:
        record = {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
            "status": "existing",
        }
    else:
        record = _copy_response(
            opener,
            url,
            destination,
            expected_size=expected_size,
        )
        record["status"] = "downloaded"
    if expected_sha256 and record["sha256"] != expected_sha256:
        raise IOError(
            f"SHA-256 mismatch for {destination}: got {record['sha256']}, "
            f"expected {expected_sha256}"
        )
    record["url"] = url
    return record


def download_spin_segmentation_sources(
    root: Path,
    *,
    force: bool = False,
    allow_missing_upstream: bool = False,
) -> Path:
    """Download the RGB source archives referenced by SPIn-NeRF MVSeg.

    The original NeRF++ ``lf_data`` Google Drive link currently returns 404.
    For the Pinecone scene, this function therefore uses a byte-addressed
    Hugging Face mirror and verifies the original archive's published LFS
    SHA-256.  The provenance exception is explicit in the manifest.
    """
    target = root / "SPIn-NeRF" / "source_images"
    target.mkdir(parents=True, exist_ok=True)

    opener = urllib.request.build_opener()
    llff_archive = target / "llff_google_drive" / "nerf_llff_data.zip"
    llff_record = _record_existing_or_download(
        opener,
        url=SPIN_LLFF_MIRROR_URL,
        destination=llff_archive,
        expected_size=SPIN_LLFF_MIRROR_SIZE,
        expected_sha256=SPIN_LLFF_MIRROR_SHA256,
        force=force,
    )
    llff_record["provenance"] = (
        "Byte-addressed mirror of nerf_llff_data.zip. The Drive folder linked "
        "by SPIn-NeRF returned no downloadable files on 2026-07-12."
    )

    # The original Fork Drive file and its parent NeRF-Supervision folder are
    # both unavailable.  Do not substitute foreground-only SPIn cutouts.  A
    # manually recovered archive is accepted only at the archived exact byte
    # size; strict mode remains blocked until it is present.
    fork_archive = target / "fork" / "fork.zip"
    if fork_archive.is_file() and fork_archive.stat().st_size == SPIN_FORK_EXPECTED_SIZE:
        fork_record: Dict[str, Any] = {
            "path": str(fork_archive),
            "bytes": fork_archive.stat().st_size,
            "sha256": _sha256(fork_archive),
            "status": "recovered_raw_archive_requires_content_and_pose_audit",
            "url": SPIN_FORK_FILE_URL,
            "formal_evaluation_ready": False,
        }
    else:
        observed_bytes = fork_archive.stat().st_size if fork_archive.is_file() else None
        fork_record = {
            "path": str(fork_archive),
            "expected_bytes": SPIN_FORK_EXPECTED_SIZE,
            "status": "missing_official_upstream_404",
            "url": SPIN_FORK_FILE_URL,
            "replacement_allowed": False,
            "formal_evaluation_ready": False,
            "reason": (
                "No verified mirror found; SPIn _cutout images are not full RGB."
            ),
        }
        if observed_bytes is not None:
            # Never delete a user-supplied recovery candidate.  Record the
            # mismatch and keep the formal cohort blocked until it is audited.
            fork_record["observed_bytes"] = observed_bytes
            fork_record["reason"] = (
                "A local recovery candidate exists but its byte size differs "
                "from the archived original; it was preserved and is not trusted."
            )

    lego_archive = (
        target
        / "lego_real_night_radial"
        / "lego_real_night_radial.tar.gz"
    )
    lego_record = _record_existing_or_download(
        opener,
        url=SPIN_LEGO_MIRROR_URL,
        destination=lego_archive,
        expected_size=SPIN_LEGO_MIRROR_SIZE,
        expected_sha256=SPIN_LEGO_MIRROR_SHA256,
        force=force,
    )
    lego_record["provenance"] = (
        "Replacement published by the LUDVIG authors in naver/ludvig issue #8; "
        "byte size matches the archived original SPIn-NeRF Drive file."
    )

    pinecone_archive = target / "nerf_real_360" / "nerf_real_360.zip"
    pinecone_record = _record_existing_or_download(
        opener,
        url=SPIN_PINECONE_MIRROR_URL,
        destination=pinecone_archive,
        expected_size=1_653_956_363,
        expected_sha256=SPIN_PINECONE_MIRROR_SHA256,
        force=force,
    )
    pinecone_record["provenance"] = (
        "Mirror of the NeRF++ lf_data archive; original Drive file "
        "1gsjDjkbTh4GAR9fFqlIDZ__qR9NYTURQ returned 404 on 2026-07-12."
    )

    tandt_archive = target / "tandt" / "tandt_db.zip"
    tandt_record = _record_existing_or_download(
        opener,
        url=SPIN_TANDT_3DGS_URL,
        destination=tandt_archive,
        expected_size=682_628_995,
        force=force,
    )

    records: List[Dict[str, Any]] = [
        llff_record,
        pinecone_record,
        tandt_record,
        lego_record,
        fork_record,
    ]
    for record in records:
        record_path = Path(record["path"])
        if record_path.is_absolute():
            record["path"] = str(record_path.relative_to(target))

    _write_manifest(
        target,
        {
            "dataset": "SPIn-NeRF multiview segmentation RGB sources",
            "downloaded_at_unix": int(time.time()),
            "source_urls": {
                "llff": SPIN_LLFF_DRIVE_URL,
                "llff_verified_mirror": SPIN_LLFF_MIRROR_URL,
                "fork": SPIN_FORK_FILE_URL,
                "lego": SPIN_LEGO_FILE_URL,
                "lego_verified_mirror": SPIN_LEGO_MIRROR_URL,
                "pinecone_mirror": SPIN_PINECONE_MIRROR_URL,
                "truck_preprocessed_3dgs": SPIN_TANDT_3DGS_URL,
            },
            "files": records,
            # An exact byte-size match is not sufficient: the recovered ZIP
            # still needs image/content validation and a protocol-locked pose
            # reconstruction before it can enter the ten-scene benchmark.
            "missing_required_assets": ["fork_verified_rgb_and_poses"],
            "complete_10_scene_rgb_cohort": False,
            "total_bytes": sum(int(record.get("bytes", 0)) for record in records),
        },
    )
    if not bool(fork_record.get("formal_evaluation_ready")) and not allow_missing_upstream:
        raise RuntimeError(
            "SPIn-NeRF Fork RGB is unavailable from its official Drive and no "
            "verified, posed recovery was found. The source manifest was written, "
            "but a formal 10-scene evaluation must remain blocked. Pass "
            "--allow-missing-upstream only to retain the audited partial download."
        )
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("dataset/promptable_nvs"),
        help="Download root (large data should normally live on /mnt/pool)",
    )
    parser.add_argument(
        "--component",
        choices=(
            "all",
            "nvos-annotations",
            "nvos-llff",
            "spin-official",
            "spin-segmentation-sources",
        ),
        default="all",
    )
    parser.add_argument("--force", action="store_true", help="Redownload existing files")
    parser.add_argument(
        "--allow-missing-upstream",
        action="store_true",
        help=(
            "Write an audited partial SPIn source manifest when a known-dead "
            "upstream asset is unavailable; never makes it 10-scene eligible"
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.component in {"all", "nvos-annotations"}:
        print(download_nvos_annotations(root, force=args.force))
    if args.component in {"all", "nvos-llff"}:
        print(download_nvos_llff(root, force=args.force))
    if args.component in {"all", "spin-official"}:
        print(download_spin_official(root, force=args.force))
    if args.component in {"all", "spin-segmentation-sources"}:
        print(
            download_spin_segmentation_sources(
                root,
                force=args.force,
                allow_missing_upstream=args.allow_missing_upstream,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
