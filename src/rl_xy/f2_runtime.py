"""Relocatable access to the declared Qwen3 f2 baseline cost model.

The historical result identities remain separate: this release changes paths,
metadata and tests, and is not presented as the historical frozen source tree.
"""
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'experiments/2026-10-29-qwen3-f2'
INPUTS = ROOT / 'experiments/2026-10-29-qwen3/inputs'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_assets():
    manifest = json.loads((ASSETS / 'assets-manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = ASSETS / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError('f2 asset missing or changed: ' + name)
    return manifest


def make_operator(model, platform, *, cache_dir=None):
    if model not in ('4B', '8B', '14B'):
        raise ValueError('unsupported f2 model')
    manifest = verify_assets()
    config = INPUTS / ('Qwen3-' + model) / 'config.json'
    if sha(config) != manifest['input_hashes'][model]['config.json']:
        raise ValueError('f2 model config differs from historical input')
    if platform == 'gpu':
        from .qwen3_cost import Operators
        op = Operators(ROOT, config, 'gpu', gpu_calibration=ASSETS / 'profiles/gpu-local.json')
        op.profile['communication_service_overhead'] = json.loads((ASSETS / 'profiles/gpu-protocol.json').read_text())
    elif platform == 'wafer':
        os.environ['RLXY_F2_REPO'] = str(ROOT)
        sys.path.insert(0, str(ASSETS / 'wafer'))
        reference = importlib.import_module('independent_timeline')
        if Path(reference.__file__).resolve().parent != (ASSETS / 'wafer').resolve():
            raise RuntimeError('conflicting f2 reference already imported')
        provider = importlib.import_module('timeline_provider')
        for name in ('timeline_provider', 'memory_scenarios', 'exact_mapper_cache'):
            module = sys.modules[name]
            if Path(module.__file__).resolve().parent != (ASSETS / 'wafer').resolve():
                raise RuntimeError('conflicting f2 module already imported: ' + name)
        op = provider.TimelineOperators(ROOT, config, 'wafer', cache_dir=cache_dir, apply_d2d_candidate=True)
    else:
        raise ValueError('platform must be gpu or wafer')
    native_binary = os.environ.get('ASTRA_SIM_BIN')
    op.runtime_cost_identity = {
        'release': 'qwen3-f2-relocatable',
        'assets_manifest_sha256': sha(ASSETS / 'assets-manifest.json'),
        'native_binary_sha256': sha(native_binary) if native_binary else None,
        'numerical_packet_bytes': 65536,
        'historical_bundle_sha256': manifest['historical_bundle_sha256'],
        'physical_end_to_end_calibrated': False,
    }
    return op


def configure_native():
    """Require an explicit built backend; do not silently select an old binary."""
    from .qwen3_paths import native_binary
    path = native_binary(ROOT)
    if not os.environ.get('ASTRA_SIM_ROOT') or not os.environ.get('ASTRA_SIM_BIN') or not path.is_file():
        raise ValueError('set ASTRA_SIM_ROOT and ASTRA_SIM_BIN to the f2 packet-streaming backend')
    sys.path.insert(0, os.environ['ASTRA_SIM_ROOT'])
    os.environ['RLXY_NETWORK_PACKET_BYTES'] = '65536'
    from . import qwen3_native
    if not getattr(qwen3_native.run_astrasim, '_rlxy_f2_watchdog', False):
        original = qwen3_native.run_astrasim
        def run(*args, **kwargs):
            kwargs['timeout_seconds'] = 7200
            return original(*args, **kwargs)
        run._rlxy_f2_watchdog = True
        qwen3_native.run_astrasim = run
    return path
