"""Sequential local stages; no training/test selection or concurrent GPU jobs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

project=Path(__file__).resolve().parents[1]/'outputs/budgeted-tool-rl'
os.chdir(project);info='../../work/model-location.json'
sys.path.insert(0,str(project))
root=Path('results/sft-v3')
print(json.dumps({'event':'waiting_for_verified_paired_analysis'}),flush=True)
while not (root/'paired-before-training.json').exists():time.sleep(5)
from toolrl.sft3_experiment import validate,paired_gate
paired_gate(validate(root/'plan.json'))
base_env=os.environ.copy();base_env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
train_env=base_env|{'PYTORCH_CUDA_ALLOC_CONF':'backend:native,garbage_collection_threshold:0.5'}
def run(*args,env=base_env):
    print(json.dumps({'event':'stage_started','args':args}),flush=True)
    subprocess.run([sys.executable,'-m',*args],check=True,env=env)
for mode,out in [('smoke','memory-probe'),('train','training')]:
    path=root/out/'manifest.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8'))['status']=='complete'
    else:run('toolrl.sft3_train',mode,'--checkpointing','--model-info',info,'--out',str(root/out),env=train_env)
run('toolrl.sft3_report','audit','--model-info',info)
run('toolrl.sft3_experiment','evaluate','--model-info',info)
run('toolrl.sft3_experiment','signal','--model-info',info)
run('toolrl.sft3_report','report','--model-info',info)
