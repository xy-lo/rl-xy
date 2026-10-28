"""Invoke the pinned official AstraSim binary without a shell or fallback."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ..types import AstraRunResult


PINNED_ASTRA_COMMIT = "4ce9ecfe746a3ee5fe47559a11d91644d641fa7f"


class AstraExecutionError(RuntimeError):
    """AstraSim could not be launched or returned a non-zero exit status."""

    def __init__(self, message: str, result: AstraRunResult | None = None) -> None:
        super().__init__(message)
        self.result = result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _astra_commit(binary: Path) -> str:
    root_text = os.environ.get("ASTRA_SIM_ROOT")
    if root_text:
        root = Path(root_text).resolve()
    else:
        # .../build/astra_analytical/build/bin/<binary> -> repository root
        try:
            root = binary.parents[4]
        except IndexError as exc:
            raise AstraExecutionError("cannot infer ASTRA_SIM_ROOT") from exc
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise AstraExecutionError(
            "cannot resolve AstraSim git commit: " + completed.stderr.strip()
        )
    commit = completed.stdout.strip()
    if commit != PINNED_ASTRA_COMMIT:
        raise AstraExecutionError(
            f"AstraSim commit mismatch: expected {PINNED_ASTRA_COMMIT}, got {commit}"
        )
    return commit


def run_astrasim(
    *,
    trace_prefix: str | Path,
    system_config: str | Path,
    network_config: str | Path,
    remote_memory_config: str | Path,
    output_dir: str | Path,
    binary: str | Path | None = None,
    timeout_seconds: int = 600,
    resource_contract: str | Path | None = None,
    network_model: str | None = None,
    comm_groups: str | Path | None = None,
) -> AstraRunResult:
    """Run AstraSim once and preserve stdout, stderr, and the debug log."""

    binary_path = Path(
        binary
        or os.environ.get(
            "ASTRA_SIM_BIN",
            "/opt/astra-sim/build/astra_analytical/build/bin/"
            "AstraSim_Analytical_Congestion_Unaware",
        )
    ).resolve()
    inputs = {
        "binary": binary_path,
        "trace_prefix": Path(trace_prefix).resolve(),
        "system_config": Path(system_config).resolve(),
        "network_config": Path(network_config).resolve(),
        "remote_memory_config": Path(remote_memory_config).resolve(),
    }
    if not binary_path.is_file() or not os.access(binary_path, os.X_OK):
        raise AstraExecutionError(f"AstraSim binary is not executable: {binary_path}")
    for label in ("system_config", "network_config", "remote_memory_config"):
        path = inputs[label]
        if not path.is_file():
            raise AstraExecutionError(f"missing {label}: {path}")
    if not Path(f"{inputs['trace_prefix']}.0.et").is_file():
        raise AstraExecutionError(
            f"missing rank-0 workload trace for prefix {inputs['trace_prefix']}"
        )

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log_dir = out / "astra_logs"
    log_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = out / "astra_stdout.log"
    stderr_path = out / "astra_stderr.log"
    debug_path = log_dir / "log.log"
    commit = _astra_commit(binary_path)
    command = (
        str(binary_path),
        f"--workload-configuration={inputs['trace_prefix']}",
        f"--system-configuration={inputs['system_config']}",
        f"--network-configuration={inputs['network_config']}",
        f"--remote-memory-configuration={inputs['remote_memory_config']}",
        f"--logging-folder={log_dir}",
    )
    if comm_groups is not None:
        groups_path=Path(comm_groups).resolve()
        if not groups_path.is_file():raise AstraExecutionError('missing communicator group config')
        command += (f'--comm-group-configuration={groups_path}',)
    (out / "astra_command.json").write_text(
        json.dumps({"argv": list(command), "exit_status": None}, indent=2) + "\n",
        encoding="utf-8",
    )
    started = _utc_now()
    environment={**os.environ, "LC_ALL": "C", "LANG": "C"}
    environment.pop("RLXY_NETWORK_TOPOLOGY",None)
    if network_model is not None:
        if network_model not in ("mesh2d","gpu_cluster") or os.environ.get("RLXY_RESOURCE_BACKEND")!="1" or binary_path.name!="AstraSim_Analytical_Congestion_Aware":
            raise AstraExecutionError("mesh2d requires the patched congestion-aware backend")
        environment["RLXY_NETWORK_TOPOLOGY"]=network_model
    if resource_contract is not None:
        if os.environ.get("RLXY_RESOURCE_BACKEND")!="1":
            raise AstraExecutionError("resource execution requires the patched rl-xy image")
        environment["RLXY_RESOURCE_CONTRACT"]=str(Path(resource_contract).resolve())
    else:
        environment.pop("RLXY_RESOURCE_CONTRACT",None)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=out,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        stdout = getattr(exc, "stdout", "") or ""
        stderr = getattr(exc, "stderr", "") or str(exc)
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        raise AstraExecutionError(f"AstraSim launch failed: {exc}") from exc
    finished = _utc_now()
    (out / 'astra_command.json').write_text(json.dumps({
        'argv': list(command), 'exit_status': completed.returncode,
        'started_at_utc': started, 'finished_at_utc': finished,
    }, indent=2) + '\n', encoding='utf-8')
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    result = AstraRunResult(
        command=command,
        return_code=completed.returncode,
        started_at_utc=started,
        finished_at_utc=finished,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        debug_log_path=debug_path,
        binary_path=binary_path,
        binary_sha256=sha256_file(binary_path),
        astra_commit=commit,
    )
    if completed.returncode != 0:
        raise AstraExecutionError(
            f"AstraSim exited with status {completed.returncode}; see {stderr_path}",
            result,
        )
    if not debug_path.is_file() or debug_path.stat().st_size == 0:
        raise AstraExecutionError("AstraSim produced no debug log", result)
    return result
