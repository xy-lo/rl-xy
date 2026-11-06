import argparse,hashlib,json,os,sys,time,traceback
from collections import defaultdict
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];sys.path[:0]=[str(R/'src'),str(ROOT/'src')]
from exact_mapper import install_exact_mapper
from baseline_contract import f2_gpu_policy
from full_method_sim import FullBackend,FullMethodSim
from rl_xy.qwen3_experiment import Policy
class Collocated(FullMethodSim):
 def __init__(self,*args,assignment,release,reward_service,**kwargs):
  super().__init__(*args,**kwargs);self.assignment=assignment;self.release_mode=release;self.reward_service=reward_service;self.group_rr=0
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
   for gid in dict.fromkeys(x.gid for x in rs):self.issue_group(self.group_reqs[gid],t)
  else:raise ValueError(self.release_mode)
class CollocatedEarly(Collocated):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.early_offloads={}
  assert self.assignment=='group_chunk' and self.release_mode=='instance'
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

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--platform',choices=['gpu','wafer'],required=True);p.add_argument('--output-root',required=True);p.add_argument('--runtime-reserve-gib',type=float,default=0);p.add_argument('--model',choices=['4B','8B','14B'],required=True);p.add_argument('--mode',choices=['verl','rlinf_collocated'],required=True);p.add_argument('--window',choices=['test1'],required=True);p.add_argument('--assignment',choices=['striped','group_chunk'],required=True);p.add_argument('--release',choices=['global','group','instance'],required=True);p.add_argument('--reward-service',choices=['parallel','group_serial'],default='parallel');p.add_argument('--prompts',type=int,default=192);p.add_argument('--run-id',required=True);a=p.parse_args()
 work=Path(a.output_root).resolve();work.mkdir(parents=True,exist_ok=True);root=ROOT;out=work/a.run_id;out.mkdir(exist_ok=False);st={'status':'RUNNING','pid':os.getpid(),'start':time.time(),'args':vars(a)};(out/'status.json').write_text(json.dumps(st,indent=2))
 try:
  install_exact_mapper();inp=R/'inputs'/f'{a.model}-test1.json';bankp=R/'inputs'/f'{a.model}-reward.jsonl';groups=[g for g in json.loads(inp.read_text()) if g['window']==a.window][:a.prompts];assert len(groups)==a.prompts
  rows=[json.loads(x) for x in bankp.read_text().splitlines()];bank={(x['group_id'],x['sample']):x for x in rows};assert len(rows)==len(bank) and all(x['status']=='PASS_MEASUREMENT' for x in rows)
  policy=Policy(**f2_gpu_policy(root,a.model));be=FullBackend(root,a.model,policy,a.platform,active_world=48,cost_contract='f2',cost_cache_dir=work/'f2-cost-cache')
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
  if a.mode=='verl':
   assert (a.assignment,a.release,a.reward_service)==('striped','global','parallel');cls=Collocated
  else:
   assert (a.assignment,a.release,a.reward_service)==('group_chunk','instance','parallel');cls=CollocatedEarly
  sim=cls(be,groups,0,'verl',a.prompts,'baseline',reward_bank=bank,released_source_training=True,assignment=a.assignment,release=a.release,reward_service=a.reward_service)
  result=sim.run();(out/'step-000.json').write_text(json.dumps(result,indent=2));ident={'model':'Qwen3-'+a.model,'mode':a.mode,'platform':a.platform,'world':48,'window':a.window,'prompts':a.prompts,'samples':a.prompts*8,'assignment':a.assignment,'release':a.release,'reward_service':a.reward_service,'policy':vars(policy),'data_sha256':sha(inp),'bank_sha256':sha(bankp),'source_sha256':{x.name:sha(x) for x in (R/'src').glob('*.py')},'runner_sha256':sha(Path(__file__)),'runtime_reserve_gib':a.runtime_reserve_gib,'cost_identity':be.op.baseline_contract_identity,'rlinf_commit':'0338bb5d85adbf0114e0b3201ea55fe2d6263ca6','scope':'48 active ranks f2 timing simulation; early per-instance offload uses the common host bandwidth cost assumption; no numerical RL'};(out/'identity.json').write_text(json.dumps(ident,indent=2))
  st.update(status='COMPLETE',end=time.time(),seconds=result['seconds'],exit_code=0);(out/'status.json').write_text(json.dumps(st,indent=2));print(json.dumps(st))
 except BaseException:
  st.update(status='FAILED',end=time.time(),error=traceback.format_exc());(out/'status.json').write_text(json.dumps(st,indent=2));raise
if __name__=='__main__':main()
