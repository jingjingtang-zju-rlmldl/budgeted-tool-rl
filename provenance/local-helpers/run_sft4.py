"""One local GPU process at a time; frozen sources/data and no cloud usage."""
import json
import os
from pathlib import Path
import subprocess
import sys

project=Path(__file__).resolve().parents[1]/'outputs/budgeted-tool-rl'
os.chdir(project);root=Path('results/sft-v4');info='../../work/model-location.json'
env=os.environ.copy();env['PYTORCH_CUDA_ALLOC_CONF']='backend:native,garbage_collection_threshold:0.5'
train_env=env|{'PYTORCH_CUDA_ALLOC_CONF':'backend:native,garbage_collection_threshold:0.5'}

def run(*args,environment=env):
    print(json.dumps(dict(event='stage_started',args=args)),flush=True)
    subprocess.run([sys.executable,'-m',*args],env=environment,check=True)

if not (root/'training-set-manifest.json').exists():
    run('toolrl.sft4_experiment','collect','--model-info',info)
    if not (root/'baseline-before-training.json').exists():run('toolrl.sft4_experiment','baseline','--model-info',info)
    run('toolrl.sft4_experiment','freeze-train','--model-info',info)
for mode,name in [('smoke','memory-probe'),('train','training')]:
    path=root/name/'manifest.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))['status']=='complete'
    else:run('toolrl.sft4_train',mode,'--checkpointing','--model-info',info,'--out',str(root/name),environment=train_env)
run('toolrl.sft4_report','audit','--model-info',info)
if not (root/'test-analysis-before-signal.json').exists():run('toolrl.sft4_experiment','evaluate','--model-info',info)
run('toolrl.sft4_experiment','signal','--model-info',info)
run('toolrl.sft4_report','report','--model-info',info)
