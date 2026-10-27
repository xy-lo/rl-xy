#!/usr/bin/env python3
"""Reconnect to the known research server through observed local SSH forwards.

No passwords are stored. New forwarded ports must present the previously
trusted server key. A remote command is submitted only once, since a lost
connection does not prove that the remote operation failed to start.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


FINGERPRINT = os.environ.get("RLXY_SSH_FINGERPRINT", "")
HOST_ALIAS = os.environ.get("RLXY_SSH_HOST_ALIAS", "")
SOCKET = os.environ.get("RLXY_SSH_SOCKET", "/tmp/rlxy-ssh.sock")
SSH_USER = os.environ.get("RLXY_SSH_USER", "")


def call(argv, timeout=20, **kwargs):
    return subprocess.run(argv, timeout=timeout, capture_output=True, text=True, **kwargs)


def observed_ports():
    r = call(["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"])
    ports = set()
    for line in r.stdout.splitlines()[1:]:
        name = line.split()[0] if line.split() else ""
        if name != "ssh" and not name.startswith("Code"):
            continue
        m = re.search(r"(?:127\.0\.0\.1|\[::1\]):(\d+)\s", line)
        if m:
            ports.add(int(m[1]))
    return sorted(ports)


def trusted_forward(port):
    try:
        scanned = call(["ssh-keyscan", "-p", str(port), "-T", "3", "-t", "ed25519", "127.0.0.1"], timeout=5)
        if not scanned.stdout:
            return False
        fp = call(["ssh-keygen", "-lf", "-"], input=scanned.stdout)
        return any(FINGERPRINT in line.split() for line in fp.stdout.splitlines())
    except (OSError, subprocess.TimeoutExpired):
        return False


def connection(port):
    return ["ssh", "-S", SOCKET, "-p", str(port), "-o", f"HostKeyAlias={HOST_ALIAS}",
            "-o", "StrictHostKeyChecking=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3", f"{SSH_USER}@localhost"]


def connect(preferred):
    if preferred:
        master = call(["ssh", "-S", SOCKET, "-O", "check", "-p", str(preferred), f"{SSH_USER}@localhost"])
        if master.returncode == 0:
            # Multiplex ignores a newly supplied -p when reusing a master.
            # Report the master's real TCP peer rather than the stale hint.
            pid = re.search(r"pid=(\d+)", master.stderr + master.stdout)
            if pid:
                sockets = call(["lsof", "-nP", "-a", "-p", pid[1], "-iTCP", "-Fn"])
                for line in sockets.stdout.splitlines():
                    peer = re.search(r"->(?:127\.0\.0\.1|\[::1\]):(\d+)$", line)
                    if peer:
                        return int(peer[1])
            raise RuntimeError("live SSH master found but its forwarded peer port cannot be identified")
        if trusted_forward(preferred):
            candidates = [preferred]
        else:
            candidates = []
    else:
        candidates = []
    if not candidates:
        ports = observed_ports()
        with ThreadPoolExecutor(max_workers=8) as pool:
            candidates = [p for p, ok in zip(ports, pool.map(trusted_forward, ports)) if ok]
    for port in candidates:
        argv = connection(port)
        r = call(argv[:-1] + ["-M", "-o", "ControlPersist=600", "-fN", argv[-1]], timeout=20)
        if r.returncode == 0:
            return port
    raise RuntimeError("no usable trusted SSH forward/key authentication; upstream VS Code/VPN channel may need restoration")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if not (FINGERPRINT and HOST_ALIAS and SSH_USER):
        p.error("set RLXY_SSH_FINGERPRINT, RLXY_SSH_HOST_ALIAS and RLXY_SSH_USER from your trusted SSH configuration")
    try:
        port = connect(a.port)
        command = a.command[1:] if a.command[:1] == ["--"] else a.command
        if not command:
            print(json.dumps({"host": "localhost", "port": port, "user": SSH_USER, "control_socket": SOCKET,
                              "server_fingerprint": FINGERPRINT, "keepalive_seconds": 20}))
            return 0
        r = subprocess.run(connection(port) + [shlex.join(command)])
        if r.returncode == 255:
            print("SSH lost: remote execution status is unknown; inspect its run ID before retrying this command", file=sys.stderr)
        return r.returncode
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"remote_session: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
