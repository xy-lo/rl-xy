"""Optional bit-exact host acceleration of the frozen FIFO event oracle."""
import ctypes as C
import hashlib,subprocess,sys,json,time,os,types
from pathlib import Path
from dataclasses import replace

class Problem(C.Structure):_fields_=[(k,C.c_int64) for k in ('k','ab','bb','ob','acc')]
class Core(C.Structure):_fields_=[(k,C.c_int64) for k in ('id','bc','mc','nc','tm','tn','tk')]
class Hardware(C.Structure):_fields_=[(k,C.c_double) for k in ('shared','load','store','gemm','vec')]+[('sram',C.c_int64)]
class Result(C.Structure):_fields_=[(k,C.c_double) for k in ('ns','busy','work')]+[(k,C.c_int64) for k in ('active','reads','writes','flops','vectors','events','tails','workspace','peak')]

def build(folder):
    source=Path(__file__).with_name('fluid.cpp');sha=hashlib.sha256(source.read_bytes()).hexdigest();folder=Path(folder)/sha;folder.mkdir(parents=True,exist_ok=True)
    flags=['-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-shared','-fPIC'];lib=folder/('fluid.dylib' if sys.platform=='darwin' else 'fluid.so')
    if not lib.exists():
        tmp=lib.with_suffix('.'+str(os.getpid())+'.tmp');subprocess.run(['clang++',*flags,str(source),'-o',str(tmp)],check=True,capture_output=True);tmp.replace(lib)
    api=C.CDLL(str(lib));fn=api.idea1_fluid_gemm;fn.argtypes=[C.POINTER(Problem),C.POINTER(Core),C.c_int,C.POINTER(Hardware),C.POINTER(Result),C.c_char_p];fn.restype=C.c_int
    return fn,dict(cpp_sha256=sha,binary_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),compiler='clang++',flags=flags,scope='host event implementation only; frozen mathematical streams and hardware rates unchanged')

def result_for(fn,p,hw,parts,load,store):
    descriptors=(Core*len(parts))(*(Core(c,ext[1],ext[3],ext[5],*mp.tile) for c,ext,mp in parts))
    timing=replace(hw,gemm_flops_per_ns=hw.gemm_fp32_flops_per_ns) if max(p.a_bytes,p.b_bytes)>=4 else hw
    h=Hardware(min(timing.dram_bytes_per_ns,timing.module_path_bytes_per_ns),load,store,timing.gemm_flops_per_ns,timing.vector_ops_per_ns,timing.sram_bytes_per_core)
    problem=Problem(p.k,p.a_bytes,p.b_bytes,p.output_bytes,p.accumulator_bytes);r=Result();err=C.create_string_buffer(512)
    code=fn(C.byref(problem),descriptors,len(parts),C.byref(h),C.byref(r),err)
    if code:raise RuntimeError('fluid backend: '+err.value.decode())
    return dict(reference_ns=r.ns,reference_memory_busy_ns=r.busy,reference_per_core_compute_max_ns=r.work,active_cores=r.active,
                audit=dict(reads=r.reads,writes=r.writes,flops=r.flops,vectors=r.vectors,events=r.events,tail_tiles=r.tails,max_workspace=r.workspace),
                peak_concurrent_private_endpoint_transfers=r.peak,sram_load_B_per_ns=load,sram_store_B_per_ns=store)

def install(op,folder,validation):
    if validation.get('status')!='PASS_EXACT_ALL_FIELDS':raise ValueError('host acceleration not validated')
    fn,identity=build(folder)
    assert validation['identity']['cpp_sha256']==identity['cpp_sha256']
    import timeline_provider as provider_module
    def compute(self,p,hw):
        parts=provider_module.capacity_partition(p,hw) if hw.cores_per_die>1 else ((0,(0,p.batches,0,p.m,0,p.n),provider_module.choose_os_mapping(p,hw.sram_bytes_per_core)),)
        result=result_for(fn,p,hw,parts,self.load,self.store)
        assert result['audit']['flops']==p.flops
        assert result['audit']['reads']==sum(mp.dram_read_bytes for _,_,mp in parts)
        assert result['audit']['writes']==sum(mp.dram_write_bytes for _,_,mp in parts)
        return result
    op.provider._compute=types.MethodType(compute,op.provider)
    op.runtime_cost_identity['host_accelerator']=identity
    return identity
