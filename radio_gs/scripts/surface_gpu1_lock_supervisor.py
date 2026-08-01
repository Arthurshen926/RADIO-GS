#!/usr/bin/env python3
"""Own the one canonical physical-GPU1 lock and supervise a runner."""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys


CANONICAL_GPU1_LOCK = Path("/root/RADIO-GS/output/.physical_gpu1.lock")
LOCK_FD_ENV = "RADIO_GS_GPU1_LOCK_FD"
LOCK_PATH_ENV = "RADIO_GS_GPU1_LOCK_PATH"
GPU1_SINGLETON_ADDRESS = b"\0radio-gs-physical-gpu1-v1"
GPU1_SINGLETON_PROTOCOL = (
    "linux-abstract-af-unix-stream-v1:radio-gs-physical-gpu1-v1"
)
SINGLETON_FD_ENV = "RADIO_GS_GPU1_SINGLETON_FD"
SINGLETON_PROTOCOL_ENV = "RADIO_GS_GPU1_SINGLETON_PROTOCOL"


def _open_canonical_lock(path: Path = CANONICAL_GPU1_LOCK) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("physical GPU1 lock requires O_NOFOLLOW")
    descriptor = os.open(
        path,
        os.O_RDWR
        | os.O_CREAT
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        info = os.fstat(descriptor)
        path_info = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(info.st_mode)
            or not stat.S_ISREG(path_info.st_mode)
            or info.st_nlink != 1
            or path_info.st_nlink != 1
            or (info.st_dev, info.st_ino)
            != (path_info.st_dev, path_info.st_ino)
        ):
            raise ValueError("physical GPU1 lock must be one regular hard link")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.set_inheritable(descriptor, True)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _singleton_protocol(address: bytes = GPU1_SINGLETON_ADDRESS) -> str:
    if not address.startswith(b"\0") or len(address) <= 1:
        raise ValueError("GPU1 singleton requires a non-empty abstract address")
    try:
        name = address[1:].decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("GPU1 singleton address must be ASCII") from error
    return f"linux-abstract-af-unix-stream-v1:{name}"


def _open_kernel_singleton(address: bytes = GPU1_SINGLETON_ADDRESS) -> int:
    """Bind a pathname-independent Linux kernel singleton until fd close."""

    if not sys.platform.startswith("linux"):
        raise RuntimeError("physical GPU1 singleton requires Linux")
    _singleton_protocol(address)
    endpoint = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        endpoint.bind(address)
        endpoint.set_inheritable(True)
        return endpoint.detach()
    except BaseException:
        endpoint.close()
        raise


def verify_inherited_singleton(
    descriptor: int,
    address: bytes = GPU1_SINGLETON_ADDRESS,
) -> dict[str, int | str]:
    if descriptor < 0:
        raise ValueError("inherited singleton descriptor must be non-negative")
    protocol = _singleton_protocol(address)
    if os.environ.get(SINGLETON_FD_ENV) != str(descriptor):
        raise ValueError("inherited physical GPU1 singleton descriptor differs")
    if os.environ.get(SINGLETON_PROTOCOL_ENV) != protocol:
        raise ValueError("inherited physical GPU1 singleton protocol differs")
    duplicate = os.dup(descriptor)
    try:
        endpoint = socket.socket(fileno=duplicate)
        with endpoint:
            if endpoint.family != socket.AF_UNIX:
                raise ValueError("inherited physical GPU1 singleton family differs")
            if endpoint.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_STREAM:
                raise ValueError("inherited physical GPU1 singleton type differs")
            if endpoint.getsockname() != address:
                raise ValueError("inherited physical GPU1 singleton address differs")
    except BaseException:
        # socket.socket(fileno=...) owns duplicate only after construction.
        try:
            os.close(duplicate)
        except OSError:
            pass
        raise
    if not os.get_inheritable(descriptor):
        raise ValueError("inherited physical GPU1 singleton is not inheritable")
    return {
        "protocol": protocol,
        "fd": descriptor,
        "socket_type": int(socket.SOCK_STREAM),
    }


def verify_inherited_lock(
    descriptor: int,
    path: Path = CANONICAL_GPU1_LOCK,
) -> dict[str, int | str]:
    if descriptor < 0:
        raise ValueError("inherited lock descriptor must be non-negative")
    info = os.fstat(descriptor)
    path_info = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISREG(info.st_mode)
        or not stat.S_ISREG(path_info.st_mode)
        or info.st_nlink != 1
        or path_info.st_nlink != 1
        or (info.st_dev, info.st_ino)
        != (path_info.st_dev, path_info.st_ino)
    ):
        raise ValueError("inherited physical GPU1 lock identity differs")
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if os.environ.get(LOCK_FD_ENV) != str(descriptor):
        raise ValueError("inherited physical GPU1 lock descriptor differs")
    if os.environ.get(LOCK_PATH_ENV) != str(path):
        raise ValueError("inherited physical GPU1 lock path differs")
    return {
        "path": str(path),
        "fd": descriptor,
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "links": int(info.st_nlink),
    }


def supervise(command: list[str]) -> int:
    if not command:
        raise ValueError("lock supervisor requires a command")
    descriptor = _open_canonical_lock()
    singleton_descriptor: int | None = None
    try:
        singleton_descriptor = _open_kernel_singleton()
        environment = os.environ.copy()
        environment[LOCK_FD_ENV] = str(descriptor)
        environment[LOCK_PATH_ENV] = str(CANONICAL_GPU1_LOCK)
        environment[SINGLETON_FD_ENV] = str(singleton_descriptor)
        environment[SINGLETON_PROTOCOL_ENV] = GPU1_SINGLETON_PROTOCOL
        completed = subprocess.run(
            command,
            env=environment,
            pass_fds=(descriptor, singleton_descriptor),
            check=False,
        )
        return int(completed.returncode)
    finally:
        if singleton_descriptor is not None:
            os.close(singleton_descriptor)
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("argv", nargs=argparse.REMAINDER)
    verify = subparsers.add_parser("verify-inherited")
    verify.add_argument("--fd", type=int, required=True)
    verify.add_argument("--singleton-fd", type=int, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "run":
        command = list(args.argv)
        if command[:1] == ["--"]:
            command = command[1:]
        raise SystemExit(supervise(command))
    record = {
        "file_lock": verify_inherited_lock(args.fd),
        "kernel_singleton": verify_inherited_singleton(args.singleton_fd),
    }
    print(record)


if __name__ == "__main__":
    main()
