"""Frozen sequential wording, correction SFT, held-out tests, and fresh signals."""
import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
from pathlib import Path
from .sft2_experiment import sha256,read,rows,write
from .sft3_data import write_splits,request_examples,encode_example
from .stress_prompts import PROMPTS

SOURCES=['sft3_data.py','sft3_train.py','sft3_episode.py','sft3_run.py','sft3_verify.py','sft3_experiment.py',
         'sft3_report.py','sft2_data.py','sft2_experiment.py','sft2_train.py','sft2_report.py',
         'stress_data.py','stress_environment.py','stress_prompts.py','environment.py','baseline.py','sft_data.py']
CONDITIONS=[('old','protocol'),('old','examples'),('new','protocol')]

def prepare(root,data,info_path):
    assert not root.exists();audit=write_splits(data);info=read(info_path)
    old=Path('results/sft-v2/training/adapter');prior=read(old.parent/'manifest.json')
    assert prior['status']=='complete'
    root.mkdir(parents=True);(root/'prompts').mkdir();prompts={}
    for name,text in PROMPTS.items():
        p=root/'prompts'/(name+'.txt');p.write_text(text,encoding='utf-8',newline='\n');prompts[name]=str(p)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(info['snapshot'],local_files_only=True)
    def encoded(split):return [encode_example(x,tok,1536) for t in rows(data/(split+'.jsonl')) for x in request_examples(t,PROMPTS['protocol'])]
    train,dev=encoded('train'),encoded('dev')
    plan=dict(prepared_at=datetime.now(timezone.utc).isoformat(),root=str(root),data_root=str(data),
        model_id=info['model_id'],model_revision=info['revision'],old_adapter_path=str(old),
        old_adapter_files_sha256=prior['adapter_files_sha256'],old_adapter_state_sha256=prior['final_adapter_state_sha256'],
        old_training_manifest_path=str(old.parent/'manifest.json'),old_training_manifest_sha256=sha256(old.parent/'manifest.json'),
        historical_files_sha256=audit['historical_files_sha256'],dataset_sha256=audit['sha256'],
        dataset_audit_sha256=sha256(data/'audit.json'),data_counts=audit['counts'],decision_targets=len(train),
        max_train_tokens=max(len(r['input_ids']) for r in train),max_dev_tokens=max(len(r['input_ids']) for r in dev),
        prompt_files=prompts,prompt_sha256={n:sha256(p) for n,p in prompts.items()},
        source_sha256={n:sha256(Path(__file__).with_name(n)) for n in SOURCES},
        training=dict(seed=42,learning_rate=1e-4,gradient_accumulation=8,epochs=1,max_context=1536,
                      rank=8,alpha=16,dropout=.05,target_modules=['q_proj','v_proj'],micro_batch=1,
                      optimizer='AdamW',weight_decay=0.,gradient_clip=1.,loss='assistant-only per-request mean'),
        memory_test=dict(optimizer_steps=8,probe_discarded=True,gradient_checkpointing=True,empty_cache_after_each_update=True,
                         allocator='backend:native,garbage_collection_threshold:0.5'),
        evaluation_seeds=dict(pair=[17,10017],dev=[17],test=[17,10017],recovery=[17,10017],signal=[30017,40017,50017,60017]),
        eval_conditions=['/'.join(x) for x in CONDITIONS],
        eval_fixed=dict(max_context=4096,max_new_tokens=64,max_calls=6,max_turns=7,temperature=.7,top_p=.8,top_k=20,
                        enable_thinking=False,batch_size=1,parser='fenced'),
        paired_design='32 facts x two exact previous signal wordings x two roots; equal sampling_index and seeds within fact; alternate wording order by root',
        correction_data='160 scripted, actually executed erroneous prefixes (80 each error type), 140 clean trajectories; bad prefix fully masked',
        held_out_test='new facts and new complete phrasings; same tool schema and bounded synthetic English environment',
        checkpoint_selection='fixed final one epoch; no test or signal based checkpoint selection',
        signal_policy='32 fresh tasks x four roots, new/protocol only; fixed old correctness-gated reward; no RL updates')
    write(root/'plan.json',plan);validate(root/'plan.json')
    print(json.dumps(dict(event='prepared',counts=plan['data_counts'],targets=len(train),max_train_tokens=plan['max_train_tokens'])),flush=True)

def validate(path):
    p=read(path)
    for n,h in p['source_sha256'].items():assert sha256(Path(__file__).with_name(n))==h,'Source changed: '+n
    for s,h in p['dataset_sha256'].items():assert sha256(Path(p['data_root'])/(s+'.jsonl'))==h
    assert sha256(Path(p['data_root'])/'audit.json')==p['dataset_audit_sha256']
    for n,h in p['prompt_sha256'].items():assert sha256(p['prompt_files'][n])==h
    for n,h in p['old_adapter_files_sha256'].items():assert sha256(Path(p['old_adapter_path'])/n)==h
    for n,h in p['historical_files_sha256'].items():assert sha256(n)==h
    assert sha256(p['old_training_manifest_path'])==p['old_training_manifest_sha256'];return p

def checkpoint(p,path):
    m=read(Path(p['root'])/'training/manifest.json')
    assert m['status']=='complete' and m['mode']=='train' and m['plan_sha256']==sha256(path)
    assert m['continuation'] and m['adapter_changed'] and m['microbatches']==p['decision_targets']
    for n,h in m['adapter_files_sha256'].items():assert sha256(Path(p['root'])/'training/adapter'/n)==h
    return m

def run_dir(p,split,model,prompt,seed):return Path(p['root'])/'eval'/f'{split}-{model}-{prompt}-seed{seed}'

def completed(p,split,model,prompt,seed):
    d=run_dir(p,split,model,prompt,seed);m=read(d/'manifest.json');v=read(d/'verification.json')
    assert m['status']=='complete' and v['episodes_sha256']==sha256(d/'episodes.jsonl')
    assert v['tasks_verified']==p['data_counts'][split];return d

def paired_gate(p):
    a=read(Path(p['root'])/'paired-before-training.json')
    assert a['plan_sha256']==sha256(Path(p['root'])/'plan.json')
    for seed in p['evaluation_seeds']['pair']:
        d=completed(p,'pair','old','protocol',seed)
        assert a['episodes_sha256'][d.name]==sha256(d/'episodes.jsonl')

def signal_gate(p):
    a=read(Path(p['root'])/'test-analysis-before-signal.json')
    assert a['plan_sha256']==sha256(Path(p['root'])/'plan.json')
    for split in ('test','recovery'):
        for seed in p['evaluation_seeds'][split]:
            for model,prompt in CONDITIONS:
                d=completed(p,split,model,prompt,seed);assert a['episodes_sha256'][d.name]==sha256(d/'episodes.jsonl')
    for seed in p['evaluation_seeds']['pair']:completed(p,'pair','new','protocol',seed)

def evaluate(p,path,info,split,model,prompt,seed):
    subprocess.run([sys.executable,'-m','toolrl.sft3_run','--plan',str(path),'--model-info',str(info),
                    '--split',split,'--model',model,'--prompt',prompt,'--seed',str(seed)],check=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','paired','evaluate','signal'])
    parser.add_argument('--root',type=Path,default=Path('results/sft-v3'));parser.add_argument('--data-root',type=Path,default=Path('data/sft-v3'))
    parser.add_argument('--model-info',type=Path,required=True);a=parser.parse_args();path=a.root/'plan.json'
    if a.stage=='prepare':prepare(a.root,a.data_root,a.model_info);return
    p=validate(path)
    if a.stage=='paired':
        for seed in p['evaluation_seeds']['pair']:evaluate(p,path,a.model_info,'pair','old','protocol',seed)
        from .sft3_report import save_paired
        save_paired(p,'old','paired-before-training.json');return
    checkpoint(p,path);paired_gate(p)
    if a.stage=='evaluate':
        for seed in p['evaluation_seeds']['pair']:evaluate(p,path,a.model_info,'pair','new','protocol',seed)
        for split in ('dev','test','recovery'):
            for seed in p['evaluation_seeds'][split]:
                for model,prompt in (CONDITIONS if seed==17 else list(reversed(CONDITIONS))):evaluate(p,path,a.model_info,split,model,prompt,seed)
        from .sft3_report import interim
        interim(p);return
    signal_gate(p)
    for seed in p['evaluation_seeds']['signal']:evaluate(p,path,a.model_info,'signal','new','protocol',seed)

if __name__=='__main__':main()
