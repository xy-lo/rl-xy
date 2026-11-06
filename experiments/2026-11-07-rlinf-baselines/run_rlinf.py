import argparse,hashlib,json,os,sys,time,traceback
from collections import defaultdict
from pathlib import Path
from dataclasses import replace
R=Path(__file__).resolve().parent;ROOT=R.parents[1];sys.path[:0]=[str(R/'src'),str(ROOT/'src')]
from exact_mapper import install_exact_mapper
from baseline_contract import f2_gpu_policy
from full_method_sim import FullBackend,FullMethodSim
from rl_xy.qwen3_experiment import Policy
class Collocated(FullMethodSim):
 def __init__(self,*args,assignment,release,reward_service,reassembly,**kwargs):
  super().__init__(*args,**kwargs);self.assignment=assignment;self.release_mode=release;self.reward_service=reward_service;self.reassembly=reassembly;self.group_rr=0
  if assignment=='group_chunk':
   for rep in self.replicas:rep.queue.clear()
   groups=list(self.group_reqs.values());q,extra=divmod(len(groups),len(self.replicas));offset=0
   for i,rep in enumerate(self.replicas):
    count=q+(i<extra)
    for group in groups[offset:offset+count]:
     for r in group:rep.queue.append(r);r.replica=i
    offset+=count
   assert offset==len(groups)
  self.rep_requests=defaultdict(list)
  for r in self.reqs.values():self.rep_requests[r.replica].append(r)
  self.emitted_replicas=set();self.emitted_groups=set()
 def issue_group(self,group,t):
  if group[0].gid in self.emitted_groups:return
  self.emitted_groups.add(group[0].gid)
  if self.reward_service=='parallel':
   for r in group:self.submit_reward(r.rid,t)
  else:
   i=self.group_rr%len(self.cpu_ready);self.group_rr+=1;start=max(t,self.cpu_ready[i]);rows=[]
   for r in group:
    assert r.rid not in self.reward_submitted;self.reward_submitted.add(r.rid)
    duration=float(self.bank[(r.gid,r.sample)]['seconds']);end=start+duration
    rows.append({'request':r.rid,'domain':'code','resource':'cpu','worker':i,'submitted':t,'start':start,'end':end,'group':r.gid});start=end
   self.cpu_ready[i]=start
   for row in rows:
    row['result_ready']=start;self.reward_tasks.append(row);self.log(t,'REWARD_SUBMIT',request=row['request'],worker=i);self.push(start,'cpu_reward_done',row['request'])
 def reward(self,rid,t):
  r=self.reqs[rid]
  if self.release_mode=='global':
   if not all(x.status=='finished' for x in self.reqs.values()) or self.baseline_reward_dispatched:return
   self.baseline_reward_dispatched=True
   for group in self.group_reqs.values():self.issue_group(group,t)
  elif self.release_mode=='group':
   group=self.group_reqs[r.gid]
   if all(x.status=='finished' for x in group):self.issue_group(group,t)
  elif self.release_mode=='instance':
   rs=self.rep_requests[r.replica]
   if r.replica in self.emitted_replicas or not all(x.status=='finished' for x in rs):return
   self.emitted_replicas.add(r.replica);self.log(t,'INSTANCE_RELEASE',replica=r.replica,requests=len(rs))
   if self.assignment=='striped':
    if self.reassembly=='partial':
     for sample in rs:self.submit_reward(sample.rid,t)
    else:
     for gid in dict.fromkeys(x.gid for x in rs):
      group=self.group_reqs[gid]
      if all(x.replica in self.emitted_replicas for x in group):
       self.log(t,'SPLIT_GROUP_REASSEMBLED',group=gid,replicas=sorted({x.replica for x in group}))
       self.issue_group(group,t)
   else:
    for gid in dict.fromkeys(x.gid for x in rs):self.issue_group(self.group_reqs[gid],t)
  elif self.release_mode=='trajectory':
   assert self.assignment=='striped' and self.reassembly=='partial'
   self.submit_reward(rid,t)
   rs=self.rep_requests[r.replica]
   if r.replica not in self.emitted_replicas and all(x.status=='finished' for x in rs):
    self.emitted_replicas.add(r.replica);self.log(t,'INSTANCE_RELEASE',replica=r.replica,requests=len(rs))
  else:raise ValueError(self.release_mode)
class CollocatedEarly(Collocated):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.early_offloads={}
  assert self.assignment in ('group_chunk','striped') and self.release_mode in ('instance','trajectory') and self.reward_service=='parallel'
 def reward(self,rid,t):
  before=set(self.emitted_replicas);super().reward(rid,t)
  for replica in sorted(self.emitted_replicas-before):
   tp=self.be.p.rollout_tp;ranks=list(range(replica*tp,(replica+1)*tp));rep=self.replicas[replica]
   assert rep.period is None and not rep.active and not rep.queue
   ends=self.transfer(ranks,lambda r:self.be.state['rollout_weights_per_rank'],'D2H',t,'RLINF_INSTANCE_ROLLOUT_OFFLOAD')
   self.early_offloads.update(ends)
 def prepare_training(self,t,workers):
  ranks=[r for w in workers for r in range(w*self.be.p.train_tp,(w+1)*self.be.p.train_tp)]
  if not ranks:return {}
  assert not self.loaded.intersection(ranks) and set(ranks)<=set(self.early_offloads)
  for rank in ranks:
   rep=self.replicas[rank//self.be.p.rollout_tp];assert rep.period is None and not rep.active
  def payload(r):
   a=self.allocs[r];return sum(a[k] for k in ('actor_BF16_parameter_buffer','master_FP32_shard','reference_BF16_valid_parameters'))
  loaded=self.transfer(ranks,payload,'H2D',max(t,max(self.early_offloads.values())),'TRAIN_ACTOR_MASTER_REFERENCE_LOAD')
  zero=max(self.allocs[r]['full_FP32_gradient_working_buffer'] for r in ranks)/(self.be.op.hw.dram_bytes_per_ns*1e9)
  self.loaded.update(ranks)
  return {w:max(loaded[r] for r in range(w*self.be.p.train_tp,(w+1)*self.be.p.train_tp))+zero for w in workers}

class ResidencyExperiment(CollocatedEarly):
 def __init__(self,*args,state_mode,control_overhead,**kwargs):
  self.state_mode=state_mode;self.control_overhead=control_overhead;self.data_tasks=[];self.pending_data=set()
  super().__init__(*args,**kwargs)
  self.resident_bytes=self.be.state['training_base_max'] if state_mode=='resident_ipc' else 0
  assert self.resident_bytes+self.be.state['rollout_weights_per_rank']+self.runtime_bytes+self.be.state['export_scratch_per_rank']<self.be.capacity
 def rollout_available(self):
  return super().rollout_available()-getattr(self,'resident_bytes',0)
 def submit_reward(self,rid,t):
  assert rid not in self.pending_data and rid not in self.reward_submitted;self.pending_data.add(rid)
  r=self.reqs[rid];rank=r.replica*self.be.p.rollout_tp;payload=4*(r.prompt+r.generated)+32
  end=self.transfer([rank],lambda _:payload,'D2H',t,'SAMPLE_TO_REWARD')[rank]
  self.data_tasks.append({'request':rid,'rank':rank,'bytes':payload,'submitted':t,'delivered':end});self.push(end,'reward_transfer_done',rid)
 def handle_event(self,t,kind,data):
  if kind=='reward_transfer_done':
   self.pending_data.remove(data);return FullMethodSim.submit_reward(self,data,t)
  return super().handle_event(t,kind,data)
 def reward(self,rid,t):
  if self.state_mode!='resident_ipc':return super().reward(rid,t)
  before=set(self.emitted_replicas);Collocated.reward(self,rid,t)
  for replica in sorted(self.emitted_replicas-before):
   tp=self.be.p.rollout_tp;ranks=list(range(replica*tp,(replica+1)*tp));rep=self.replicas[replica]
   assert rep.period is None and not rep.active and not rep.queue
   start=max(t,max(self.host_ready[rank//8] for rank in ranks));end=start+self.control_overhead
   self.log(start,'ROLL_ENGINE_DISCARD_AND_SLEEP',replica=replica,ranks=ranks,end=end,bytes_d2h=0)
   self.early_offloads.update({rank:end for rank in ranks})
 def prepare_training(self,t,workers):
  if self.state_mode!='resident_ipc':return super().prepare_training(t,workers)
  ranks=[r for w in workers for r in range(w*self.be.p.train_tp,(w+1)*self.be.p.train_tp)]
  if not ranks:return {}
  assert not self.loaded.intersection(ranks) and set(ranks)<=set(self.early_offloads)
  for rank in ranks:
   rep=self.replicas[rank//self.be.p.rollout_tp];assert rep.period is None and not rep.active
  zero=max(self.allocs[r]['full_FP32_gradient_working_buffer'] for r in ranks)/(self.be.op.hw.dram_bytes_per_ns*1e9)
  ready=max(t,max(self.early_offloads.values()))+zero;self.loaded.update(ranks);self.log(ready,'RESIDENT_TRAIN_READY',ranks=len(ranks));return {w:ready for w in workers}
 def optimizer_ready(self,t):
  if self.state_mode=='resident_ipc':return t
  return super().optimizer_ready(t)
 def publish_ready(self,t):
  if self.state_mode=='resident_ipc':
   self.log(t,'ROLL_ENGINE_WAKE_ALLOCATE',end=t+self.control_overhead);return t+self.control_overhead
  return super().publish_ready(t)
 def finish_state(self,t):
  if self.state_mode=='resident_ipc':return t
  return super().finish_state(t)
 def decorate_summary(self,d):
  d=super().decorate_summary(d);d['reward_data_transfers']=self.data_tasks;d['state_mode']=self.state_mode;d['control_overhead_seconds']=self.control_overhead
  if self.state_mode=='resident_ipc':
   d['resident_training_bytes_per_rank']=self.resident_bytes;d['generation_resident_peak_bytes']+=self.resident_bytes
   assert d['generation_resident_peak_bytes']<=self.be.capacity*.85+1
   d['state_boundary']='actor, master, reference and Adam reside GPU; rollout sleep discards stale buffers; updated weights exported via existing GPU weight plan'
   d['state_lifetime']='GPU resident across update; model includes full generation residency and GPU weight sync; no numerical correctness verification'
  return d

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--platform',choices=['gpu','wafer'],required=True);p.add_argument('--output-root',required=True);p.add_argument('--runtime-reserve-gib',type=float,default=0);p.add_argument('--model',choices=['4B','8B','14B'],required=True);p.add_argument('--train-tp',type=int,required=True);p.add_argument('--rollout-tp',type=int,required=True);p.add_argument('--train-mb',type=int,required=True);p.add_argument('--logprob-mb',type=int,required=True);p.add_argument('--cap',type=int,required=True);p.add_argument('--dp-balance',choices=['off','source'],required=True);p.add_argument('--balance-control-overhead',type=float,default=0.5);p.add_argument('--training-assignment',choices=['input','arrival_block','arrival_rr'],required=True);p.add_argument('--reference-swap',choices=['none','cpu_model'],required=True);p.add_argument('--window',choices=['test1'],required=True);p.add_argument('--assignment',choices=['striped','group_chunk'],required=True);p.add_argument('--release',choices=['instance','trajectory'],required=True);p.add_argument('--reassembly',choices=['group_barrier','partial'],required=True);p.add_argument('--reward-service',choices=['parallel','group_serial'],default='parallel');p.add_argument('--state-mode',choices=['original','resident_ipc'],required=True);p.add_argument('--control-overhead',type=float,default=0.0);p.add_argument('--prompts',type=int,default=192);p.add_argument('--run-id',required=True);a=p.parse_args()
 work=Path(a.output_root).resolve();work.mkdir(parents=True,exist_ok=True);root=ROOT;out=work/a.run_id;out.mkdir(exist_ok=False);st={'status':'RUNNING','pid':os.getpid(),'start':time.time(),'args':vars(a)};(out/'status.json').write_text(json.dumps(st,indent=2))
 try:
  install_exact_mapper();inp=R/'inputs'/f'{a.model}-test1.json';bankp=R/'inputs'/f'{a.model}-reward.jsonl';groups=[g for g in json.loads(inp.read_text()) if g['window']==a.window][:a.prompts];assert len(groups)==a.prompts
  rows=[json.loads(x) for x in bankp.read_text().splitlines()];bank={(x['group_id'],x['sample']):x for x in rows};assert len(rows)==len(bank) and all(x['status']=='PASS_MEASUREMENT' for x in rows)
  policy=replace(Policy(**f2_gpu_policy(root,a.model)),train_tp=a.train_tp,rollout_tp=a.rollout_tp,train_mb=a.train_mb,logprob_mb=a.logprob_mb,cap=a.cap);be=FullBackend(root,a.model,policy,a.platform,active_world=48,cost_contract='f2',cost_cache_dir=work/'f2-cost-cache')
  if a.runtime_reserve_gib:
   reserve=int(a.runtime_reserve_gib*1024**3)
   def fit_chunks(lengths,limit,training):
    chunks=[];i=0
    while i<len(lengths):
     n=min(limit,len(lengths)-i)
     while be.memory(lengths[i:i+n],training)+reserve>be.capacity:
      if n==1:raise ValueError('sample plus runtime reserve exceeds capacity')
      n=max(1,n//2)
     chunks.append((i,i+n));i+=n
    return chunks
   be.fit_chunks=fit_chunks
  sim=ResidencyExperiment(be,groups,0,'verl',a.prompts,'baseline',reward_bank=bank,released_source_training=True,assignment=a.assignment,release=a.release,reward_service=a.reward_service,reassembly=a.reassembly,state_mode=a.state_mode,control_overhead=a.control_overhead)
  sim.training_assignment=a.training_assignment;sim.reference_swap=a.reference_swap=='cpu_model';sim.dp_balance=a.dp_balance=='source';sim.balance_control_overhead=a.balance_control_overhead
  result=sim.run();(out/'step-000.json').write_text(json.dumps(result,indent=2));ident={'model':'Qwen3-'+a.model,'platform':a.platform,'world':48,'window':a.window,'prompts':a.prompts,'samples':a.prompts*8,'assignment':a.assignment,'dp_balance':a.dp_balance,'balance_control_overhead':a.balance_control_overhead,'training_assignment':a.training_assignment,'reference_swap':a.reference_swap,'state_mode':a.state_mode,'control_overhead_seconds':a.control_overhead,'reassembly':a.reassembly,'release':a.release,'reward_service':a.reward_service,'policy':vars(policy),'data_sha256':sha(inp),'bank_sha256':sha(bankp),'source_sha256':{x.name:sha(x) for x in (R/'src').glob('*.py')},'runner_sha256':sha(Path(__file__)),'runtime_reserve_gib':a.runtime_reserve_gib,'cost_identity':be.op.baseline_contract_identity,'rlinf_commit':'0338bb5d85adbf0114e0b3201ea55fe2d6263ca6','scope':'48 active ranks f2 timing simulation of grouped per-instance release, input or non-clairvoyant arrival-based DP assignment; optional CPU reference swap; no numerical RL'};(out/'identity.json').write_text(json.dumps(ident,indent=2))
  st.update(status='COMPLETE',end=time.time(),seconds=result['seconds'],exit_code=0);(out/'status.json').write_text(json.dumps(st,indent=2));print(json.dumps(st))
 except BaseException:
  st.update(status='FAILED',end=time.time(),error=traceback.format_exc());(out/'status.json').write_text(json.dumps(st,indent=2));raise
if __name__=='__main__':main()
