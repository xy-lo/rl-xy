"""Load the accepted f2 local-cost assets; never silently use peak costs.

The event-level routed-ring executor is a separate, explicitly declared
approximation to the historical packet executor. Sharing local assets does
not imply equality of end-to-end historical step times.
"""
import hashlib,json
from pathlib import Path
HISTORICAL_F2_BUNDLE='2f52a646a0ac71b6e071f8eb4622e3e67142922534fe47588f8694d6a2093df4'

def make_f2_operator(root,model,platform,cache_dir=None):
    from rl_xy import f2_runtime
    root=Path(root).resolve()
    if root!=f2_runtime.ROOT.resolve():
        raise ValueError('f2 cost import root differs from requested cost root')
    manifest=f2_runtime.verify_assets()
    if manifest['historical_bundle_sha256']!=HISTORICAL_F2_BUNDLE:
        raise ValueError('cost assets do not identify the accepted f2 bundle')
    op=f2_runtime.make_operator(model,platform,cache_dir=cache_dir)
    if platform=='gpu':
        if model != '32B' and not getattr(op,'calibration_identity',None):raise ValueError('f2 GPU local calibration not active')
        if not op.profile.get('communication_service_overhead'):raise ValueError('f2 GPU protocol residual not active')
    elif not getattr(op,'local_timing_identity',None):
        raise ValueError('f2 wafer event-timeline provider not active')
    path=f2_runtime.ASSETS/'assets-manifest.json'
    op.baseline_contract_identity={'contract':'32B-analytic-GPU-unverified' if model=='32B' and platform=='gpu' else 'accepted-f2-local-costs','historical_bundle_sha256':HISTORICAL_F2_BUNDLE,
        'assets_manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'provider_class':type(op.provider).__module__+'.'+type(op.provider).__name__,
        'gpu_calibration':getattr(op,'calibration_identity',None),'wafer_local_timing':getattr(op,'local_timing_identity',None),
        'd2d_latency_ns':op.hw.d2d_latency_ns,
        'network_executor':'isolated TP routed-ring projection, not historical full packet replay'}
    return op

def f2_gpu_policy(root,model):
    if model == '32B':
        # Declared initial candidate, not a measured or optimized policy.
        return {'train_tp':8,'rollout_tp':8,'train_mb':1,'logprob_mb':1,'cap':32}
    root=Path(root);assets=root/'experiments/2026-10-29-qwen3-f2'
    manifest=json.loads((assets/'assets-manifest.json').read_text());name=f'policies/{model}-gpu.json';path=assets/name
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['files'][name]:raise ValueError('f2 GPU policy asset changed')
    if manifest['historical_bundle_sha256']!=HISTORICAL_F2_BUNDLE:raise ValueError('wrong f2 policy bundle')
    return json.loads(path.read_text())
