"""Identity-checked immutable per-batch operator reuse, never duration reuse."""
import gzip,json,hashlib,copy,os
from pathlib import Path
from rl_xy.qwen3_experiment import training_groups,chunks,teacher_lengths
from rl_xy.qwen3_sequence import compile_blocks
from rl_xy.f2_runtime import ROOT as COST_ROOT,sha
from .context import digest

class FrozenBatchCatalog:
    def __init__(self,evidence,requests,policy):
        self.policy=policy;self.entries={};self.proofs={};self.hits=0;self.misses=0
        evidence=Path(evidence);metadata=json.loads((evidence/'baseline-metadata.json').read_text())
        canonical=digest(requests);assert metadata['result_fields']['trajectory_sha256']==canonical
        assert metadata['result_fields']['policy']==policy.__dict__
        self.checked={}
        for name,expected in metadata['cost_source_files'].items():
            if name.split('/')[0] in ('src','configs','docker') and name!='src/rl_xy/qwen3_source.py':
                assert sha(COST_ROOT/name)==expected,name;self.checked[name]=expected
        groups=training_groups(requests,2,54)
        for stage in ('old_policy','reference','training'):
            path=evidence/(stage+'-program.json.gz')
            with gzip.open(path,'rt') as f:value=json.load(f)
            self.proofs[stage]=dict(path=str(path),sha256=sha(path))
            for w,items in enumerate(groups):
                segments=value['programs'][value['ranks'][str(2*w)]]['segments']
                if stage=='training':self.zero=segments[0]['steps'];segments=segments[1:]
                batches=chunks(items,policy.train_mb if stage=='training' else policy.logprob_mb)
                assert len(batches)==len(segments)
                for batch,segment in zip(batches,segments):
                    assert segment['repeat']==1
                    body=segment['steps'];assert all(x['kind'] in ('local','AR') and not set(x)&{'src','dst','ranks','group'} for x in body)
                    key=(stage,tuple(x['trajectory_id'] for x in batch));assert key not in self.entries;self.entries[key]=body
    def remap(self,groups):
        frozen={stage:{} for stage in ('old_policy','reference','training')}
        for w,items in enumerate(groups):
            for stage in frozen:
                values=[]
                if stage=='training':values.append(dict(steps=self.zero))
                for batch in chunks(items,self.policy.train_mb if stage=='training' else self.policy.logprob_mb):
                    key=(stage,tuple(x['trajectory_id'] for x in batch))
                    if key not in self.entries:raise ValueError('frozen batch is not an identical sample tuple')
                    values.append(dict(steps=self.entries[key]));self.hits+=1
                frozen[stage][w]=values
        return frozen
    def receipt(self):return dict(files=self.proofs,checked_source_files=self.checked,hits=self.hits,misses=self.misses,scope='identical per-batch native operators only; placement and all native execution are new')

class OperatorBatchCatalog:
    def __init__(self,op,identity,folder,frozen=None):
        self.op=op;self.identity=identity;self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True);self.frozen=frozen;self.hits=0;self.misses=0;self.memory={}
    def body(self,stage,tp,batch):
        tids=tuple(x['trajectory_id'] for x in batch)
        if tp==2 and self.frozen and (stage,tids) in self.frozen.entries:
            self.hits+=1;return copy.deepcopy(self.frozen.entries[stage,tids])
        lengths=teacher_lengths(batch);training=stage=='training'
        key=digest(dict(identity=self.identity,training=training,tp=tp,lengths=lengths,response_tokens=sum(x['response_length'] for x in batch) if training else None))
        if key in self.memory:self.hits+=1;return copy.deepcopy(self.memory[key])
        path=self.folder/(key+'.json')
        if path.exists():
            v=json.loads(path.read_text());assert v['key']==key and v['sha256']==digest(v['steps']);steps=v['steps'];self.hits+=1
        else:
            blocks=self.op.training(lengths,tp,sum(x['response_length'] for x in batch)) if training else self.op.forward(lengths,tp)
            steps=compile_blocks(self.op,tuple(range(tp)),blocks)
            assert all(x['kind'] in ('local','AR') and not set(x)&{'src','dst','ranks','group'} for x in steps)
            v=dict(key=key,sha256=digest(steps),steps=steps);tmp=path.with_suffix('.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(v,separators=(',',':'))+'\n');tmp.replace(path);self.misses+=1
        self.memory[key]=steps;return copy.deepcopy(steps)
