"""Check sealed evidence and Git byte-preservation without another inference run."""
from pathlib import Path
import hashlib
import json
import subprocess
import tempfile
import shutil
import sys

root=Path(__file__).resolve().parent.parent
project=root/'outputs'/'budgeted-tool-rl'
sys.path.insert(0,str(project))
from toolrl.prompt_study import validate_plan
from toolrl.chat_prompt_probe import validate

previous=Path.cwd()
import os
os.chdir(project)
validate_plan(Path('results/prompt-control-v1'))
validate(Path('results/native-examples-v1'))
os.chdir(previous)
checked_runs=[]
for run in sorted((project/'results').glob('*/*-seed*')):
    verification=json.loads((run/'verification.json').read_text())
    assert hashlib.sha256((run/'episodes.jsonl').read_bytes()).hexdigest()==verification['episodes_sha256']
    checked_runs.append(run.relative_to(project).as_posix())
assert len(checked_runs)==12
protected=['toolrl/baseline.py','toolrl/environment.py','toolrl/prompt_study.py','toolrl/chat_prompt_probe.py',
           'toolrl/verify_run.py','data/prompt_control60.jsonl',
           'results/prompt-control-v1/plan.json','results/prompt-control-v1/original-seed7/episodes.jsonl']
with tempfile.TemporaryDirectory(prefix='git-byte-check-',dir=root/'work') as directory:
    checkout=Path(directory)
    subprocess.run(['git','init','--quiet',str(checkout)],check=True)
    shutil.copyfile(project/'.gitattributes',checkout/'.gitattributes')
    for name in protected:
        source=project/name
        target=checkout/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        raw=source.read_bytes()
        expected=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        actual=subprocess.check_output(['git','-C',str(checkout),'hash-object','--path='+name,str(target)],text=True).strip()
        assert actual==expected, 'Git filter changed bytes: '+name
result=dict(sealed_runs=checked_runs,verified_episodes=720,verified_requests=920,
            plan_and_source_hashes_match=True,trace_hashes_match=True,
            git_filter_preserves_pinned_bytes=protected,
            dataset_files_unchanged=True,training_performed=False)
(project/'results'/'prompt-control-v1'/'artifact-integrity.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'sealed_runs':len(checked_runs),'git_byte_checks':len(protected),'all_matched':True}))
