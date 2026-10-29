"""Freeze public length replay inputs, official chat template, and paired windows."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,random
from collections import defaultdict,Counter
from pathlib import Path
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--inputs',type=Path,required=True,help='Staging copy containing pinned configs/tokenizers and original manifest')
 ap.add_argument('--lengths-csv',type=Path,required=True)
 ap.add_argument('--ds1000',type=Path,required=True)
 a=ap.parse_args();out=a.inputs.resolve()
 ds=[json.loads(x) for x in gzip.open(a.ds1000,'rt')]
 byid={int(x['metadata']['problem_id']):x for x in ds};assert len(byid)==1000
 rng=random.Random(20261028);pools=defaultdict(list)
 for i in sorted(byid):pools[byid[i]['metadata']['library']].append(i)
 for k in sorted(pools):rng.shuffle(pools[k])
 windows=[]
 for window in range(5):
  total=sum(map(len,pools.values()));raw={k:len(v)*192/total for k,v in pools.items()};quota={k:int(v) for k,v in raw.items()}
  for k in sorted(raw,key=lambda k:(-(raw[k]-quota[k]),k))[:192-sum(quota.values())]:quota[k]+=1
  ids=[]
  for k in sorted(pools):ids.extend(pools[k][:quota[k]]);pools[k]=pools[k][quota[k]:]
  rng.shuffle(ids);windows.append({'name':'tune' if window==0 else f'test{window}','prompt_ids':ids,'samples':{str(i):sorted(rng.sample(range(100),8)) for i in ids},'library_counts':dict(Counter(byid[i]['metadata']['library'] for i in ids))})
 selected={i for x in windows for i in x['prompt_ids']};assert len(selected)==960
 source=a.lengths_csv
 rows={}
 for r in csv.DictReader(source.open()):
  if 'ds1000-think-temp06' not in r['partition']:continue
  key=(r['model'],int(r['group_id']),int(r['sample']))
  assert key not in rows;rows[key]=r
 assert len(rows)==300000
 env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
 env.globals['raise_exception']=lambda x:(_ for _ in ()).throw(ValueError(x))
 modelman=[]
 identities=json.loads((out/'manifest.json').read_text())['models']
 for size in ['4B','8B','14B']:
  name=f'Qwen3-{size}';d=out/name;cfg=json.loads((d/'tokenizer_config.json').read_text());tok=Tokenizer.from_file(str(d/'tokenizer.json'))
  template=env.from_string(cfg['chat_template']);prompts={};lengths={}
  for i in sorted(selected):
   text=template.render(messages=[{'role':'user','content':byid[i]['prompt']}],add_generation_prompt=True,enable_thinking=True,tools=None,bos_token=cfg.get('bos_token'),eos_token=cfg.get('eos_token'))
   p=len(tok.encode(text,add_special_tokens=False).ids)
   assert 0<p and p+8192<=40960,(name,i,p)
   prompts[str(i)]={'prompt_length':p,'library':byid[i]['metadata']['library'],'rendered_sha256':hashlib.sha256(text.encode()).hexdigest()}
   lengths[i]=p
  for window in windows:
   tr=[]
   for i in window['prompt_ids']:
    for s in window['samples'][str(i)]:
     r=rows[name,i,s];n=int(r['response_length']);assert 0<n<=8192
     tr.append({'trajectory_id':f'{i}:{s}','group_id':str(i),'sample':s,'prompt_length':lengths[i],'response_length':n,'finish_reason':r['finish_reason']})
   assert len(tr)==1536
   dump(d/(window['name']+'.json'),{'model':name,'window':window['name'],'prompt_count':192,'responses_per_prompt':8,'trajectories':tr,'length_source':'recorded n_total_tokens; prompt reconstructed using pinned official chat template','full_response_tokens':sum(t['response_length'] for t in tr)})
  dump(d/'prompt_lengths.json',prompts)
  assert (d/'config.json').is_file(), 'Pinned model config missing'
  modelman.append({'model':name,'identity':next(x['identity'] for x in identities if x['model']==name),'files':{p.name:sha(p) for p in sorted(d.iterdir()) if p.is_file()},'prompt_length_min':min(lengths.values()),'prompt_length_max':max(lengths.values())})
 dump(out/'windows.json',{'seed':20261028,'windows':windows,'reserved_prompt_ids':sorted(set(byid)-selected)})
 dump(out/'manifest.json',{'status':'PASS','seed':20261028,'sampling':'library-stratified disjoint prompt windows, equal batch1536; fixed sample IDs across models and platforms','dataset_revision':'a726a001d0145eddb2d5a2f5ede60591950b141f','lengths_csv_sha256':sha(source),'ds1000_gzip_sha256':sha(a.ds1000),'windows_sha256':sha(out/'windows.json'),'models':modelman,'numeric_answers_regenerated':False,'input_template_reconstructed':True})
 print(json.dumps({'status':'PASS','models':len(modelman),'windows_per_model':5,'trajectories_per_window':1536,'prompt_length_range':[(m['model'],m['prompt_length_min'],m['prompt_length_max']) for m in modelman]}),flush=True)
if __name__=='__main__':main()
