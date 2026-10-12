"""Frozen collection rule, pre-SFT controls, one-epoch training, blind signal."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .sft4_data import write_splits, request_examples, encode_example, harvest
from .sft4_prompts import PROMPTS

SOURCES=['sft4_environment.py','sft4_prompts.py','sft4_data.py','sft4_episode.py','sft4_verify.py','sft4_run.py',
         'sft4_train.py','sft4_experiment.py','sft4_report.py','environment.py','baseline.py','sft_data.py']
CONDITIONS=[('old','protocol'),('old','examples'),('new','protocol')]
def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines()]
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
def now():return datetime.now(timezone.utc).isoformat()


def prepare(root,data,info_path):
    assert not root.exists(); a=write_splits(data); info=read(info_path)
    old=Path('results/sft-v3/training/adapter'); prior=read(old.parent/'manifest.json');assert prior['status']=='complete'
    root.mkdir(parents=True); (root/'prompts').mkdir(); prompts={}
    for name,text in PROMPTS.items():
        f=root/'prompts'/(name+'.txt'); f.write_text(text,encoding='utf-8',newline='\n'); prompts[name]=str(f)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(info['snapshot'],local_files_only=True)
    train=[encode_example(x,tok,2048) for t in rows(data/'train.jsonl') for x in request_examples(t,PROMPTS['protocol'])]
    p=dict(prepared_at=now(),root=str(root),data_root=str(data),model_id=info['model_id'],model_revision=info['revision'],
        old_adapter_path=str(old),old_adapter_files_sha256=prior['adapter_files_sha256'],old_adapter_state_sha256=prior['final_adapter_state_sha256'],
        old_training_manifest_path=str(old.parent/'manifest.json'),old_training_manifest_sha256=sha256(old.parent/'manifest.json'),
        historical_files_sha256=a['historical_files_sha256'],dataset_sha256=a['sha256'],dataset_audit_sha256=sha256(data/'audit.json'),
        data_counts=a['counts'],clean_decision_targets=len(train),clean_max_train_tokens=max(len(x['input_ids']) for x in train),
        prompt_files=prompts,prompt_sha256={n:sha256(f) for n,f in prompts.items()},
        source_sha256={n:sha256(Path(__file__).with_name(n)) for n in SOURCES},
        training=dict(seed=42,learning_rate=1e-4,gradient_accumulation=8,epochs=1,max_context=2048,
                      rank=8,alpha=16,dropout=.05,target_modules=['q_proj','v_proj'],micro_batch=1,optimizer='AdamW',
                      weight_decay=0.,gradient_clip=1.,loss='assistant-only per-request mean'),
        memory_test=dict(optimizer_steps=8,probe_discarded=True,gradient_checkpointing=True,empty_cache_after_each_update=True,
                         allocator='backend:native,garbage_collection_threshold:0.5'),
        evaluation_seeds=dict(collect=[27,10027],reserve=[20027,30027],dev=[17],test=[17,10017],recovery=[17,10017],
                              signal=[30017,40017,50017,60017]),
        collection_order=[['collect',27],['reserve',20027],['collect',10027],['reserve',30027]],
        collection_rule='Stop after first cohort yielding >=8 unique eligible natural-copy facts; at most four fixed cohorts. First eligible actual unknown_entity single-edit mistake per fact; unique known-ID repair; legal public-state completion within original 6/7 budget. Sort by cohort order then sampling_index; cap 80 distinct facts. No synthetic prefixes in training.',
        clean_training_tasks=240,natural_target_facts=8,natural_cap=80,eval_conditions=['/'.join(c) for c in CONDITIONS],
        eval_fixed=dict(max_context=4096,max_new_tokens=96,max_calls=6,max_turns=7,temperature=.7,top_p=.8,top_k=20,
                        enable_thinking=False,batch_size=1,parser='fenced'),
        reward=dict(primary='grounded_correct * (1 - 0.02 * weighted_query_cost)',ordinary_query_cost=1,
                    resolved_limit_cost='question-quoted 2 or 6, including errors',max_possible_cost=36,
                    auxiliary='exact_correct * (1 - 0.03 * call_count)',changed_from_v3=True),
        oracle='Only supplied facts and actual observations; no hidden-policy peek. Price-2 service dominates unknown-policy routes; price-6 uses observed branch; known prefixes compare remaining cost.',
        checkpoint_selection='fixed final one epoch, no test/signal selection',
        signal_policy='32 fresh fact/wording tasks x four fixed roots, new/protocol only; never train on diagnostic set',
        candidate_rl_gate=dict(min_nonconstant_groups=8,min_distinct_kinds=3,
            meaning='Predeclared practical screening gate, not a theorem or guarantee; next step still requires new training-task validation and RL memory measurement'),
        correction_recovery='32 scripted single-edit states; edit x chain position x wording fully crossed. Separate from actual natural-copy training sources.')
    write(root/'plan.json',p); validate(root/'plan.json')
    print(json.dumps(dict(event='prepared',counts=p['data_counts'],clean_targets=len(train),clean_max_tokens=p['clean_max_train_tokens'])),flush=True)


def validate(path):
    p=read(path)
    for name,h in p['source_sha256'].items():assert sha256(Path(__file__).with_name(name))==h,'Frozen source changed: '+name
    for s,h in p['dataset_sha256'].items():assert sha256(Path(p['data_root'])/(s+'.jsonl'))==h
    assert sha256(Path(p['data_root'])/'audit.json')==p['dataset_audit_sha256']
    for n,h in p['prompt_sha256'].items():assert sha256(p['prompt_files'][n])==h
    for n,h in p['old_adapter_files_sha256'].items():assert sha256(Path(p['old_adapter_path'])/n)==h
    assert sha256(p['old_training_manifest_path'])==p['old_training_manifest_sha256']
    for n,h in p['historical_files_sha256'].items():assert sha256(n)==h
    return p


def run_dir(p,split,model,prompt,seed):return Path(p['root'])/'eval'/f'{split}-{model}-{prompt}-seed{seed}'
def completed(p,split,model,prompt,seed):
    d=run_dir(p,split,model,prompt,seed); m=read(d/'manifest.json'); v=read(d/'verification.json')
    assert m['status']=='complete' and v['episodes_sha256']==sha256(d/'episodes.jsonl') and v['tasks_verified']==p['data_counts'][split]
    return d


def build_train(p):
    clean=rows(Path(p['data_root'])/'train.jsonl'); selected=[]; seen=set(); provenance={}; order=[]
    for split,seed in p['collection_order']:
        d=run_dir(p,split,'old','protocol',seed)
        if not (d/'verification.json').exists(): break
        completed(p,split,'old','protocol',seed); order.append([split,seed])
        raw={r['task_id']:r for r in rows(d/'episodes.jsonl')}
        tasks=sorted(rows(Path(p['data_root'])/(split+'.jsonl')),key=lambda t:t['sampling_index'])
        provenance[d.name]=sha256(d/'episodes.jsonl')
        for t in tasks:
            if t['fact_group'] in seen:continue
            derived=harvest(t,raw[t['task_id']])
            if derived:
                derived['source_cohort']=d.name; derived['source_episodes_sha256']=provenance[d.name]
                selected.append(derived); seen.add(t['fact_group'])
        if len(selected)>=p['natural_target_facts']:break
    assert order # collect before training
    selected=selected[:p['natural_cap']]
    return clean+selected,selected,provenance,order


def baseline_gate(p):
    a=read(Path(p['root'])/'baseline-before-training.json');assert a['plan_sha256']==sha256(Path(p['root'])/'plan.json')
    for split in ('dev','test','recovery'):
        for seed in p['evaluation_seeds'][split]:
            for model,prompt in CONDITIONS[:2]:
                d=completed(p,split,model,prompt,seed);assert a['episodes_sha256'][d.name]==sha256(d/'episodes.jsonl')


def training_gate(p):
    baseline_gate(p); root=Path(p['root']); c=read(root/'training-set-manifest.json')
    assert c['plan_sha256']==sha256(root/'plan.json') and sha256(Path(p['data_root'])/'derived_train.jsonl')==c['derived_train_sha256']
    actual,selected,provenance,order=build_train(p)
    assert actual==rows(Path(p['data_root'])/'derived_train.jsonl') and provenance==c['source_episodes_sha256'] and order==c['collection_order_used']
    assert len(selected)==c['natural_training_tasks']>0 and len(actual)==c['training_tasks']
    return c


def freeze_train(p,info_path):
    baseline_gate(p); root=Path(p['root']); assert not (root/'training-set-manifest.json').exists()
    actual,selected,provenance,order=build_train(p);assert selected,'No actual eligible copy error; cannot claim correction SFT'
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(read(info_path)['snapshot'],local_files_only=True)
    prompt=Path(p['prompt_files']['protocol']).read_text(encoding='utf-8')
    enc=[encode_example(x,tok,p['training']['max_context']) for t in actual for x in request_examples(t,prompt)]
    from .sft4_data import save_rows
    path=Path(p['data_root'])/'derived_train.jsonl'; assert not path.exists(); save_rows(path,actual)
    from collections import Counter
    c=dict(frozen_at=now(),plan_sha256=sha256(root/'plan.json'),derived_train_sha256=sha256(path),
        training_tasks=len(actual),clean_training_tasks=p['clean_training_tasks'],natural_training_tasks=len(selected),
        decision_targets=len(enc),max_train_tokens=max(len(x['input_ids']) for x in enc),
        natural_copy_types=dict(Counter(t['natural_copy_type'] for t in selected)),source_episodes_sha256=provenance,
        collection_order_used=order,bad_actions_and_observations_are_masked=True,actual_student_errors_not_scripted=True,
        selection_rule='predeclared eligible source rule, training facts only; no test/signal adaptation')
    write(root/'training-set-manifest.json',c); training_gate(p)
    print(json.dumps(dict(event='training_data_frozen',**{k:c[k] for k in ('training_tasks','natural_training_tasks','decision_targets','max_train_tokens','natural_copy_types')})),flush=True)


def checkpoint(p,path):
    m=read(Path(p['root'])/'training/manifest.json'); c=training_gate(p)
    assert m['status']=='complete' and m['mode']=='train' and m['plan_sha256']==sha256(path)
    assert m['continuation'] and m['adapter_changed'] and m['microbatches']==c['decision_targets']
    assert m['training_set_manifest_sha256']==sha256(Path(p['root'])/'training-set-manifest.json')
    for n,h in m['adapter_files_sha256'].items():assert sha256(Path(p['root'])/'training/adapter'/n)==h
    return m


def signal_gate(p):
    root=Path(p['root']); a=read(root/'test-analysis-before-signal.json'); assert a['plan_sha256']==sha256(root/'plan.json')
    for split in ('dev','test','recovery'):
        for seed in p['evaluation_seeds'][split]:
            for model,prompt in CONDITIONS:
                d=completed(p,split,model,prompt,seed); assert a['episodes_sha256'][d.name]==sha256(d/'episodes.jsonl')


def evaluate(p,path,info,split,model,prompt,seed):
    subprocess.run([sys.executable,'-m','toolrl.sft4_run','--plan',str(path),'--model-info',str(info),
                    '--split',split,'--model',model,'--prompt',prompt,'--seed',str(seed)],check=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','collect','baseline','freeze-train','evaluate','signal'])
    parser.add_argument('--root',type=Path,default=Path('results/sft-v4'));parser.add_argument('--data-root',type=Path,default=Path('data/sft-v4'))
    parser.add_argument('--model-info',type=Path,required=True);a=parser.parse_args();path=a.root/'plan.json'
    if a.stage=='prepare':prepare(a.root,a.data_root,a.model_info);return
    p=validate(path)
    if a.stage=='collect':
        for split,seed in p['collection_order']:
            evaluate(p,path,a.model_info,split,'old','protocol',seed)
            _,selected,_,_=build_train(p)
            print(json.dumps(dict(event='natural_copy_collection',unique_eligible_facts=len(selected),cohort=[split,seed])),flush=True)
            if len(selected)>=p['natural_target_facts']:break
        return
    if a.stage=='baseline':
        for split in ('dev','test','recovery'):
            for seed in p['evaluation_seeds'][split]:
                for model,prompt in (CONDITIONS[:2] if seed==17 else list(reversed(CONDITIONS[:2]))):evaluate(p,path,a.model_info,split,model,prompt,seed)
        from .sft4_report import save_baseline
        save_baseline(p);return
    if a.stage=='freeze-train':freeze_train(p,a.model_info);return
    checkpoint(p,path)
    if a.stage=='evaluate':
        for split in ('dev','test','recovery'):
            for seed in p['evaluation_seeds'][split]:evaluate(p,path,a.model_info,split,'new','protocol',seed)
        from .sft4_report import save_interim
        save_interim(p);return
    signal_gate(p)
    for seed in p['evaluation_seeds']['signal']:evaluate(p,path,a.model_info,'signal','new','protocol',seed)

if __name__=='__main__':main()
