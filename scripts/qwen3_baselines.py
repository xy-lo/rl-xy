"""Screen or replay a Qwen3 candidate using explicit portable input/output paths."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rl_xy.qwen3_cost import Operators
from rl_xy.qwen3_experiment import Screen,Policy,run_candidate
from rl_xy.qwen3_paths import input_root,output_path


def main():
 root=Path(__file__).resolve().parents[1]
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('action',choices=['screen','candidate'])
 p.add_argument('--inputs',type=Path,default=input_root(root))
 p.add_argument('--model',choices=['4B','8B','14B'],required=True)
 p.add_argument('--platform',default='gpu',choices=['gpu','wafer'])
 p.add_argument('--window',default='tune',choices=['tune','test1','test2','test3','test4'])
 p.add_argument('--policy',type=Path)
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();out=output_path(root,a.output);inputs=a.inputs.resolve()/f'Qwen3-{a.model}'
 if a.action=='candidate' and a.policy is None:p.error('candidate requires --policy')
 op=Operators(root,inputs/'config.json',a.platform)
 requests=json.loads((inputs/(a.window+'.json')).read_text())['trajectories']
 if a.action=='screen':Screen(op,requests).run(out)
 else:
  policy=Policy(**json.loads(a.policy.read_text()));result=run_candidate(op,requests,policy,out)
  print(json.dumps({'status':result['status'],'model':a.model,'platform':a.platform,'window':a.window,'step_s':result['step_model_ns']/1e9,'wall_s':result['wall_seconds']}),flush=True)

if __name__=='__main__':main()
