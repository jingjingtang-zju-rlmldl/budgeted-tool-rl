"""Conditional extra training collection, then original fixed controls and signal."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

project=Path(__file__).resolve().parents[1]/'outputs/budgeted-tool-rl';os.chdir(project);sys.path.insert(0,str(project))
root=Path('results/sft-v4');info='../../work/model-location.json'
from toolrl.sft4_copy_rules import amendment,extended_build,ORIGINAL_BUILD
from toolrl.sft4_experiment import validate
amendment();p=validate(root/'plan.json')
assert not ORIGINAL_BUILD(p)[1]
print(json.dumps(dict(event='waiting_for_original_baseline')),flush=True)
while not (root/'baseline-before-training.json').exists():time.sleep(5)
env=os.environ.copy();env['PYTORCH_CUDA_ALLOC_CONF']='backend:native,garbage_collection_threshold:0.5'
train_env=env|{'PYTORCH_CUDA_ALLOC_CONF':'backend:native,garbage_collection_threshold:0.5'}
def run(*args,environment=env):
    print(json.dumps(dict(event='stage_started',args=args)),flush=True)
    subprocess.run([sys.executable,'-m',*args],env=environment,check=True)
for seed in amendment()['seeds']:
    run('toolrl.sft4_copy_probe','--model-info',info,'--model','old','--prompt','protocol','--split','copy_probe','--seed',str(seed))
    _,selected,_,_=extended_build(p)
    print(json.dumps(dict(event='copy_probe_eligible_facts',count=len(selected))),flush=True)
    if len(selected)>=p['natural_target_facts']:break
if not (root/'training-set-manifest.json').exists():run('toolrl.sft4_copy_rules','dispatch','toolrl.sft4_experiment','freeze-train','--model-info',info)
for mode,name in [('smoke','memory-probe'),('train','training')]:
    path=root/name/'manifest.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))['status']=='complete'
    else:run('toolrl.sft4_copy_rules','dispatch','toolrl.sft4_train',mode,'--checkpointing','--model-info',info,'--out',str(root/name),environment=train_env)
run('toolrl.sft4_copy_rules','dispatch','toolrl.sft4_report','audit','--model-info',info)
if not (root/'test-analysis-before-signal.json').exists():run('toolrl.sft4_copy_rules','dispatch','toolrl.sft4_experiment','evaluate','--model-info',info)
run('toolrl.sft4_copy_rules','dispatch','toolrl.sft4_experiment','signal','--model-info',info)
run('toolrl.sft4_copy_rules','dispatch','toolrl.sft4_report','report','--model-info',info)
