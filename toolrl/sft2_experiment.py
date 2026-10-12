"""Freeze continuation training, fresh controls and a separate rollout probe."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

from .baseline import sha256 as file_sha
from .sft2_data import write_splits,request_examples,encode_example
from .stress_prompts import PROMPTS

SOURCES=['sft2_data.py','sft2_train.py','sft2_run.py','sft2_experiment.py',
         'stress_environment.py','stress_data.py','stress_prompts.py','stress_runner.py',
         'stress_verify.py','sft_data.py','sft_train.py','baseline.py','environment.py']
CONDITIONS=[('base','examples'),('old','protocol'),('old','examples'),('new','protocol')]


def sha256(path):return file_sha(Path(path))


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines()]
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')


def prepare(root,data,info_path):
    if root.exists():raise ValueError('New experiment root required')
    audit=write_splits(data)
    info=read(info_path)
    old=Path('results/sft-v1/training/adapter')
    prior=read(old.parent/'manifest.json')
    root.mkdir(parents=True);(root/'prompts').mkdir()
    paths={}
    for name,prompt in PROMPTS.items():
        path=root/'prompts'/(name+'.txt')
        path.write_text(prompt,encoding='utf-8',newline='\n');paths[name]=str(path)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(info['snapshot'],local_files_only=True)
    examples=[encode_example(x,tok,1024) for t in rows(data/'train.jsonl') for x in request_examples(t,PROMPTS['protocol'])]
    dev=[encode_example(x,tok,1024) for t in rows(data/'dev.jsonl') for x in request_examples(t,PROMPTS['protocol'])]
    assert len(examples)==990 and len({x['task_id'] for x in examples})==300
    plan=dict(prepared_at=datetime.now(timezone.utc).isoformat(),root=str(root),data_root=str(data),
        model_id=info['model_id'],model_revision=info['revision'],
        old_adapter_path=str(old),old_adapter_files_sha256=prior['adapter_files_sha256'],
        old_adapter_state_sha256=prior['final_adapter_state_sha256'],
        old_training_manifest_path=str(old.parent/'manifest.json'),old_training_manifest_sha256=sha256(old.parent/'manifest.json'),
        historical_files_sha256=audit['historical_files_sha256'],dataset_sha256=audit['sha256'],
        dataset_audit_sha256=sha256(data/'audit.json'),data_counts=dict(train=300,dev=27,test=72,signal=32),
        decision_targets=len(examples),max_train_tokens=max(len(x['input_ids']) for x in examples),
        max_dev_tokens=max(len(x['input_ids']) for x in dev),prompt_files=paths,
        prompt_sha256={n:sha256(p) for n,p in paths.items()},
        source_sha256={n:sha256(Path(__file__).with_name(n)) for n in SOURCES},
        training=dict(seed=42,learning_rate=1e-4,gradient_accumulation=8,epochs=1,max_context=1024,
                      rank=8,alpha=16,dropout=.05,target_modules=['q_proj','v_proj'],micro_batch=1,
                      optimizer='AdamW',weight_decay=0.,gradient_clip=1.,loss='assistant-only per-request mean'),
        memory_test=dict(optimizer_steps=8,longest_contexts=8,probe_discarded=True,
                         fallback='non-reentrant gradient checkpointing only on CUDA OOM'),
        evaluation_seeds=dict(dev=[7],test=[7,10007,20007],signal=[30007,40007,50007,60007]),
        eval_conditions=['/'.join(c) for c in CONDITIONS],
        eval_fixed=dict(max_context=4096,max_new_tokens=64,max_calls=6,max_turns=7,
                        temperature=.7,top_p=.8,top_k=20,enable_thinking=False,batch_size=1,parser='fenced'),
        base_prompt='fixed previous dev-selected examples; no new test prompt search',
        previous_selection_path='results/stress-v1/selection.json',
        previous_selection_sha256=sha256('results/stress-v1/selection.json'),
        checkpoint_selection='fixed final one epoch; no test or signal based checkpoint selection',
        signal_policy='new/protocol only, four rollouts per task, no optimizer updates',
        signal_compositions=['chain_three','chain_four','retry_three','retry_four','redirect_three','redirect_four','prefix_retry','prefix_redirect'])
    write(root/'plan.json',plan)
    validate(root/'plan.json')
    print(json.dumps(dict(event='prepared',counts=plan['data_counts'],targets=len(examples),
                         max_train_tokens=plan['max_train_tokens'],max_dev_tokens=plan['max_dev_tokens'])),flush=True)


def validate(path):
    p=read(path)
    for name,h in p['source_sha256'].items():assert sha256(Path(__file__).with_name(name))==h,'Source changed: '+name
    for split,h in p['dataset_sha256'].items():assert sha256(Path(p['data_root'])/(split+'.jsonl'))==h
    assert sha256(Path(p['data_root'])/'audit.json')==p['dataset_audit_sha256']
    for name,h in p['prompt_sha256'].items():assert sha256(p['prompt_files'][name])==h
    for name,h in p['old_adapter_files_sha256'].items():assert sha256(Path(p['old_adapter_path'])/name)==h
    for name,h in p['historical_files_sha256'].items():assert sha256(name)==h
    assert sha256(p['old_training_manifest_path'])==p['old_training_manifest_sha256']
    assert sha256(p['previous_selection_path'])==p['previous_selection_sha256']
    return p


def checkpoint(plan,path):
    manifest_path=Path(plan['root'])/'training'/'manifest.json'
    m=read(manifest_path)
    assert m['status']=='complete' and m['mode']=='train' and m['plan_sha256']==sha256(path)
    assert m['continuation'] and m['adapter_changed'] and m['microbatches']==plan['decision_targets']
    for name,h in m['adapter_files_sha256'].items():assert sha256(manifest_path.parent/'adapter'/name)==h
    return m


def evaluate(plan_path,info,split,model,prompt,seed):
    print(json.dumps(dict(event='condition_started',split=split,model=model,prompt=prompt,seed=seed)),flush=True)
    subprocess.run([sys.executable,'-m','toolrl.sft2_run','--plan',str(plan_path),'--model-info',str(info),
                    '--split',split,'--model',model,'--prompt',prompt,'--seed',str(seed)],check=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','evaluate','signal'])
    parser.add_argument('--root',type=Path,default=Path('results/sft-v2'))
    parser.add_argument('--data-root',type=Path,default=Path('data/sft-v2'))
    parser.add_argument('--model-info',type=Path,required=True)
    args=parser.parse_args();path=args.root/'plan.json'
    if args.stage=='prepare':prepare(args.root,args.data_root,args.model_info);return
    p=validate(path);checkpoint(p,path)
    if args.stage=='evaluate':
        for split in ('dev','test'):
            for seed in p['evaluation_seeds'][split]:
                order=CONDITIONS if seed!=10007 else list(reversed(CONDITIONS))
                for model,prompt in order:evaluate(path,args.model_info,split,model,prompt,seed)
    else:
        for seed in p['evaluation_seeds']['test']:
            for model,prompt in CONDITIONS:
                run=args.root/'eval'/f'test-{model}-{prompt}-seed{seed}'
                m=read(run/'manifest.json');v=read(run/'verification.json')
                assert m['status']=='complete' and v['episodes_sha256']==sha256(run/'episodes.jsonl')
        for seed in p['evaluation_seeds']['signal']:evaluate(path,args.model_info,'signal','new','protocol',seed)


if __name__=='__main__':main()
