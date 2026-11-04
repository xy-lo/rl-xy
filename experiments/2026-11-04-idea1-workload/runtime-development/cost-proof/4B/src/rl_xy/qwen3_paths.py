"""Portable Qwen3 inputs, native backend and generated-output locations."""
import os
from pathlib import Path
from contextlib import contextmanager


def input_root(root):
    return Path(root).resolve() / 'experiments/2026-10-29-qwen3/inputs'


def native_root(root):
    return Path(os.environ.get('ASTRA_SIM_ROOT', str(Path(root) / 'native/astra-sim'))).expanduser().resolve()


def native_binary(root):
    return Path(os.environ.get('ASTRA_SIM_BIN', str(native_root(root) / 'build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware'))).expanduser().resolve()


def output_path(root, value):
    root = Path(root).resolve(); value = Path(value).expanduser().resolve()
    if value == root or value in root.parents:
        raise ValueError('Generated output must not be the repository or an ancestor')
    for name in ['src', 'scripts', 'tests', 'configs', 'docker', 'docs', 'experiments', 'calibration', '.git']:
        protected = root / name
        if value == protected or protected in value.parents:
            raise ValueError('Generated output must be outside source and frozen evidence: ' + name)
    return value


def generation_cache(root):
    return output_path(root, os.environ.get('RLXY_QWEN3_CACHE_ROOT', str(Path(root) / 'runs/qwen3/generation_cache')))


@contextmanager
def campaign_lock(path):
    """Hold a nonblocking OS lock; a persistent lock file avoids unlink races."""
    import fcntl
    import json
    with Path(path).open('a+') as stream:
        try: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Campaign already active; refusing duplicate writer') from error
        try:
            stream.seek(0); stream.truncate()
            json.dump({'pid': os.getpid()}, stream); stream.flush()
            yield
        finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
