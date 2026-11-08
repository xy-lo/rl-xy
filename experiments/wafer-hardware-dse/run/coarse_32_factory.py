from pathlib import Path
from rl_xy.f2_runtime import make_operator,ROOT

def make32(model,platform,*,cache_dir=None):
    assert model=='32B' and platform=='wafer'
    ref=make_operator('14B',platform,cache_dir=cache_dir)
    from timeline_provider import TimelineOperators
    config=Path('coarse-probe/32b-tp16-replicated-config.json').resolve()
    op=TimelineOperators(ROOT,config,'wafer',cache_dir=cache_dir,apply_d2d_candidate=True)
    op.runtime_cost_identity={**ref.runtime_cost_identity,'approximation':'32B TP16 with replicated KV heads represented as 16 logical KV heads; tied gradients approximated'}
    return op
