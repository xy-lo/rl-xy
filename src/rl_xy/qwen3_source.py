"""Content identity for every public-input/native execution result."""
from pathlib import Path
import hashlib,json,tarfile,io,subprocess,platform,sys
from .qwen3_paths import native_root


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()


def manifest(root):
 root=Path(root);files={}
 for top in ['src','scripts','tests','configs','docker']:
  for p in sorted((root/top).rglob('*')):
   if p.is_file() and '__pycache__' not in p.parts and (p.suffix in ['.py','.sh','.hh','.patch','.json','.yml','.proto'] or p.name.startswith('Dockerfile')):
    files[str(p.relative_to(root))]=sha(p)
 for name in ['pyproject.toml','.gitignore']:files[name]=sha(root/name)
 canonical=json.dumps(files,sort_keys=True,separators=(',',':')).encode()
 return {'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'tree_sha256':hashlib.sha256(canonical).hexdigest(),'files':files,'python':sys.version,'host_platform':platform.platform()}


def freeze(root,out):
 root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=False);m=manifest(root)
 (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 with tarfile.open(out/'source.tar.gz','w:gz') as tar:
  for name in sorted(m['files']):
   p=root/name;info=tar.gettarinfo(str(p),arcname=name);info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
   with p.open('rb') as f:tar.addfile(info,f)
 up=native_root(root)
 patch=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=up)
 (out/'native-combined.patch').write_bytes(patch)
 (out/'native-submodules.txt').write_text(subprocess.check_output(['git','submodule','status','--recursive'],cwd=up,text=True))
 (out/'python-packages.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
 return m
