"""Private-container adapter around the user's existing veRL length collector.

Keep unrounded timing_raw and resolved algorithm config in addition to L1.
Never activates or edits the host's shared collector flag.
"""
import importlib.util
import json
import os
from pathlib import Path

ORIGINAL=Path(os.environ['RLXY_LEGACY_HOOK'])
spec=importlib.util.spec_from_file_location('rlxy_original_capture',ORIGINAL)
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
original.FLAG_PATH='/root/RLInfra/scripts/rltrace_enabled.json'


def dump_step(**kwargs):
    original.dump_step(**kwargs)
    config=json.loads(Path(original.FLAG_PATH).read_text())
    batch=kwargs['batch']
    from omegaconf import OmegaConf
    out=Path(config['out_dir'])/f'execution_step_{kwargs["global_steps"]}.json'
    report={'step':kwargs['global_steps'],'timing_raw_s':{k:float(v) for k,v in kwargs['timing_raw'].items()},
        'resolved_training_config':OmegaConf.to_container(kwargs['config'],resolve=True),
        'physical_tensor_shapes':{k:list(v.shape) for k,v in batch.batch.items()},
        'scope':'framework runtime capture; GPU step timing, not chip timing'}
    with out.open('x') as stream:json.dump(report,stream,indent=2)
