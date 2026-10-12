"""Freeze the stress suite, select the strongest base prompt on dev, then test."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

from .baseline import sha256
from .stress_data import write_splits,KINDS
from .stress_prompts import PROMPTS
from .stress_environment import MAX_CALLS,MAX_TURNS,StressEnvironment

SOURCES=['stress_environment.py','stress_data.py','stress_prompts.py','stress_runner.py',
         'stress_verify.py','stress_experiment.py','baseline.py','environment.py']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')


def prepare(root,data_root,info_path):
    if root.exists():
        raise ValueError('new experiment directory required')
    audit=write_splits(data_root)
    info=read(info_path)
    training_path=Path('results/sft-v1/training/manifest.json')
    training=read(training_path)
    adapter=training_path.parent/'adapter'
    assert training['status']=='complete' and training['optimizer_steps']==75
    assert training['model_id']==info['model_id'] and training['model_revision']==info['revision']
    for name,sha in training['adapter_files_sha256'].items():
        assert sha256(adapter/name)==sha
    root.mkdir(parents=True)
    (root/'prompts').mkdir()
    paths={}
    for name,prompt in PROMPTS.items():
        path=root/'prompts'/(name+'.txt')
        path.write_text(prompt,encoding='utf-8',newline='\n')
        paths[name]=str(path)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(info['snapshot'],local_files_only=True)
    token_counts={}
    for name,prompt in PROMPTS.items():
        counts=[]
        for split in ('dev','test'):
            for task in rows(data_root/(split+'.jsonl')):
                messages=[{'role':'system','content':prompt},{'role':'user','content':task['prompt']}]
                env=StressEnvironment(task)
                for call in task['oracle_calls']+[None]:
                    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    counts.append(len(tokenizer(text,add_special_tokens=False)['input_ids']))
                    if call:
                        action={'action':'lookup',**call}
                        observation=env.lookup(call['entity'],call['field'])
                        messages+=[{'role':'assistant','content':json.dumps(action,separators=(',',':'))},
                                   {'role':'tool','content':json.dumps(observation,sort_keys=True)}]
        token_counts[name]=max(counts)
    assert max(token_counts.values())<=4096
    plan=dict(prepared_at=datetime.now(timezone.utc).isoformat(),root=str(root),data_root=str(data_root),
              model_id=info['model_id'],model_revision=info['revision'],
              adapter_path=str(adapter),adapter_sha256=training['adapter_files_sha256']['adapter_model.safetensors'],
              adapter_files_sha256=training['adapter_files_sha256'],training_manifest_path=str(training_path),
              training_manifest_sha256=sha256(training_path),dataset_sha256=audit['sha256'],
              dataset_audit_sha256=sha256(data_root/'audit.json'),historical_files_sha256=audit['historical_files_sha256'],
              data_counts={'dev':24,'test':64},fact_groups={'dev':9,'test':24},kinds=KINDS,
              prompt_files=paths,prompt_sha256={k:sha256(Path(v)) for k,v in paths.items()},
              source_sha256={name:sha256(Path(__file__).with_name(name)) for name in SOURCES},
              evaluation_seeds={'dev':[7],'test':[7,10007,20007]},
              eval_fixed=dict(do_sample=True,temperature=.7,top_p=.8,top_k=20,enable_thinking=False,
                  dtype='bfloat16',device='cuda:0',batch_size=1,cpu_threads=4,parser='fenced',
                  max_context=4096,max_new_tokens=64,max_calls=MAX_CALLS,max_turns=MAX_TURNS),
              oracle_context_max_tokens=token_counts,
              dev_conditions=['base/protocol','base/examples','sft/protocol','sft/examples'],
              base_prompt_selection='dev exact successes, then strict successes, then fewer correct extra calls, then shorter prompt; test is never used',
              test_conditions='base/dev-selected prompt; sft/protocol; sft/dev-selected base prompt if different',
              primary_comparison='Existing SFT with extended short protocol versus base with dev-selected strongest of two prompts',
              paired_scenarios='anchor/retry/redirect/skip_owner/given_answer share facts; chain_four/skip_prefix share facts',
              seed_rule='root seed + 100*sampling_index, then + turn; paired fact-group scenarios share sampling_index',
              training_performed=False,limitations=['Controlled synthetic English schema; new cost_center and spending_limit fields.',
                  '64 test scenarios from 24 fact groups, repeated under three sampling seeds; observations are correlated.',
                  'This tests transfer of the existing adapter, not newly trained multi-hop or recovery behavior.'])
    write(root/'plan.json',plan)
    print(json.dumps(dict(event='prepared',tasks=plan['data_counts'],fact_groups=plan['fact_groups'],
                         max_oracle_context_tokens=token_counts)),flush=True)


def validate(plan_path):
    plan=read(plan_path)
    for name,sha in plan['source_sha256'].items():
        assert sha256(Path(__file__).with_name(name))==sha,'source changed: '+name
    for split,sha in plan['dataset_sha256'].items():
        assert sha256(Path(plan['data_root'])/(split+'.jsonl'))==sha
    assert sha256(Path(plan['data_root'])/'audit.json')==plan['dataset_audit_sha256']
    for path,sha in plan['historical_files_sha256'].items():
        assert sha256(Path(path))==sha
    for name,sha in plan['prompt_sha256'].items():
        assert sha256(Path(plan['prompt_files'][name]))==sha
    for name,sha in plan['adapter_files_sha256'].items():
        assert sha256(Path(plan['adapter_path'])/name)==sha
    assert sha256(Path(plan['training_manifest_path']))==plan['training_manifest_sha256']
    return plan


def simple_metrics(values):
    correct=[r for r in values if r['success']]
    return dict(tasks=len(values),successes=len(correct),strict_successes=sum(r['strict_success'] for r in values),
                correct_extra_calls=sum(r['correct_extra_calls'] for r in correct),
                by_kind={kind:sum(r['success'] for r in values if r['kind']==kind) for kind in KINDS})


def evaluate(plan_path,info,split,model,prompt,seed):
    plan=validate(plan_path)
    out=Path(plan['root'])/'eval'/f'{split}-{model}-{prompt}-seed{seed}'
    print(json.dumps(dict(event='condition_started',split=split,model=model,prompt=prompt,seed=seed)),flush=True)
    subprocess.run([sys.executable,'-m','toolrl.stress_runner','--plan',str(plan_path),'--model-info',str(info),
                    '--model',model,'--prompt',prompt,'--split',split,'--seed',str(seed)],check=True)
    subprocess.run([sys.executable,'-m','toolrl.stress_verify',str(out),'--plan',str(plan_path),'--model-info',str(info)],
                   check=True,capture_output=True,text=True)
    metrics=simple_metrics(rows(out/'episodes.jsonl'))
    write(out/'metrics.json',metrics)
    print(json.dumps(dict(event='condition_complete',split=split,model=model,prompt=prompt,seed=seed,**metrics)),flush=True)


def select_base_prompt(plan_path):
    plan=validate(plan_path)
    root=Path(plan['root'])
    candidates=[]
    provenance={}
    for prompt in ('protocol','examples'):
        path=root/'eval'/f'dev-base-{prompt}-seed7'
        manifest=read(path/'manifest.json')
        verify=read(path/'verification.json')
        assert manifest['status']=='complete' and verify['episodes_sha256']==sha256(path/'episodes.jsonl')
        values=rows(path/'episodes.jsonl')
        m=simple_metrics(values)
        candidates.append(dict(prompt=prompt,**m,prompt_characters=len(Path(plan['prompt_files'][prompt]).read_text(encoding='utf-8'))))
        provenance[prompt]=dict(episodes_sha256=sha256(path/'episodes.jsonl'),verification_sha256=sha256(path/'verification.json'))
    winner=max(candidates,key=lambda c:(c['successes'],c['strict_successes'],-c['correct_extra_calls'],-c['prompt_characters']))
    selection=dict(selected_at=datetime.now(timezone.utc).isoformat(),plan_sha256=sha256(plan_path),
                   selected_prompt=winner['prompt'],candidates=candidates,dev_provenance=provenance,
                   rule=plan['base_prompt_selection'],test_used=False)
    path=root/'selection.json'
    if path.exists():
        prior=read(path)
        assert all(prior[k]==selection[k] for k in selection if k!='selected_at')
    else:
        write(path,selection)
    print(json.dumps(dict(event='base_prompt_selected',prompt=winner['prompt'],candidates=candidates)),flush=True)


def allowed_test_conditions(plan,plan_path):
    selection=read(Path(plan['root'])/'selection.json')
    assert selection['plan_sha256']==sha256(plan_path) and not selection['test_used']
    for prompt,provenance in selection['dev_provenance'].items():
        path=Path(plan['root'])/'eval'/f'dev-base-{prompt}-seed7'
        assert sha256(path/'episodes.jsonl')==provenance['episodes_sha256']
        assert sha256(path/'verification.json')==provenance['verification_sha256']
    prompt=selection['selected_prompt']
    assert prompt in ('protocol','examples')
    conditions=[('base',prompt),('sft','protocol')]
    if prompt!='protocol':
        conditions.append(('sft',prompt))
    return conditions


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','dev','test'])
    parser.add_argument('--root',type=Path,default=Path('results/stress-v1'))
    parser.add_argument('--data-root',type=Path,default=Path('data/stress-v1'))
    parser.add_argument('--model-info',type=Path,required=True)
    args=parser.parse_args()
    plan_path=args.root/'plan.json'
    if args.stage=='prepare':
        prepare(args.root,args.data_root,args.model_info)
    elif args.stage=='dev':
        for model,prompt in [('base','protocol'),('base','examples'),('sft','protocol'),('sft','examples')]:
            evaluate(plan_path,args.model_info,'dev',model,prompt,7)
        select_base_prompt(plan_path)
    else:
        plan=validate(plan_path)
        conditions=allowed_test_conditions(plan,plan_path)
        for seed in plan['evaluation_seeds']['test']:
            order=conditions if seed!=10007 else list(reversed(conditions))
            for model,prompt in order:
                evaluate(plan_path,args.model_info,'test',model,prompt,seed)


if __name__=='__main__':
    main()
