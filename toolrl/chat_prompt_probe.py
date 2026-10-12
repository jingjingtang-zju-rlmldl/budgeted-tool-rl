"""Exploratory follow-up: three demonstrations as native conversation messages.

Prepared after inspecting the first inline-example seed; separate from the
preregistered three-condition study. No current-episode facts enter examples.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from . import baseline
from .prompt_study import NEUTRAL, SEEDS, digest, paired, diagnostics

SYSTEM = NEUTRAL + '''
The preceding example exchanges are separate completed episodes, not facts
for the latest user question. Solve the latest user question using its own
entity IDs and actual tool results. Output only the next JSON action.
'''
DEMOS = [
    {'role':'user','content':'The budget of project_demo_417f is 321. What is its budget?'},
    {'role':'assistant','content':'{"action":"answer","value":321}'},
    {'role':'user','content':'What is the budget of project_demo_62c8?'},
    {'role':'assistant','content':'{"action":"lookup","entity":"project_demo_62c8","field":"budget"}'},
    {'role':'tool','content':'{"entity":"project_demo_62c8","field":"budget","value":875}'},
    {'role':'assistant','content':'{"action":"answer","value":875}'},
    {'role':'user','content':'What department does the owner of project_demo_19a4 belong to?'},
    {'role':'assistant','content':'{"action":"lookup","entity":"project_demo_19a4","field":"owner"}'},
    {'role':'tool','content':'{"entity":"project_demo_19a4","field":"owner","value":"employee_demo_8bd9"}'},
    {'role':'assistant','content':'{"action":"lookup","entity":"employee_demo_8bd9","field":"department"}'},
    {'role':'tool','content':'{"entity":"employee_demo_8bd9","field":"department","value":"department_48216"}'},
    {'role':'assistant','content':'{"action":"answer","value":"department_48216"}'},
]


def prefix_hash():
    return hashlib.sha256(json.dumps(DEMOS,sort_keys=True).encode()).hexdigest()


def prepare(root,primary,info):
    if (root/'plan.json').exists():
        raise ValueError('follow-up already prepared')
    plan=json.loads((primary/'plan.json').read_text())
    model_info=json.loads(info.read_text())
    assert model_info['model_id']==plan['fixed']['model'] and model_info['revision']==plan['fixed']['model_revision']
    tasks=[json.loads(x) for x in Path(plan['dataset']).read_text().splitlines()]
    for t in tasks:
        for entity in t['database']:
            assert entity not in json.dumps(DEMOS)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(json.loads(info.read_text())['snapshot'],local_files_only=True)
    from .environment import LookupEnvironment
    lengths=[]
    for task in tasks:
        messages=[{'role':'system','content':SYSTEM}]+deepcopy(DEMOS)+[{'role':'user','content':task['prompt']}]
        env=LookupEnvironment(task)
        lengths.append(len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)))
        for call in task['oracle_calls']:
            messages.append({'role':'assistant','content':json.dumps({'action':'lookup',**call})})
            messages.append({'role':'tool','content':json.dumps(env.lookup(call['entity'],call['field']),sort_keys=True)})
            lengths.append(len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)))
    assert max(lengths)<1024
    root.mkdir(parents=True,exist_ok=True)
    followup=dict(purpose='exploratory follow-up after first inline-example seed, not preregistered primary comparison',
                  rationale='Inline examples improved JSON but showed example-answer copying and early termination; test native role-based examples.',
                  prepared_at=baseline.datetime.now(baseline.timezone.utc).isoformat(),
                  primary_root=str(primary),primary_plan_sha256=digest(primary/'plan.json'),
                  dataset=plan['dataset'],dataset_sha256=plan['dataset_sha256'],seeds=SEEDS,tasks=60,total_episodes=180,
                  model_id=plan['fixed']['model'],model_revision=plan['fixed']['model_revision'],
                  source_sha256=digest(Path(__file__)),baseline_source_sha256=digest(Path(baseline.__file__)),
                  environment_source_sha256=digest(Path(baseline.__file__).with_name('environment.py')),
                  verifier_source_sha256=digest(Path(__file__).with_name('verify_run.py')),
                  system_prompt_sha256=hashlib.sha256(SYSTEM.encode()).hexdigest(),demonstrations_sha256=prefix_hash(),
                  min_context_tokens=min(lengths),max_oracle_context_tokens=max(lengths),
                  comparison_limit='Changes role representation and example-boundary wording; not a perfect single-factor ablation. Runs after primary conditions; speed comparison can be confounded by desktop load.')
    (root/'plan.json').write_text(json.dumps(followup,indent=2)+'\n',encoding='utf-8')
    (root/'demonstrations.json').write_text(json.dumps(DEMOS,indent=2)+'\n',encoding='utf-8')
    (root/'system.txt').write_text(SYSTEM,encoding='utf-8',newline='\n')
    print(json.dumps({'event':'followup_prepared','planned_episodes':180,'max_context':max(lengths)}),flush=True)


def validate(root):
    plan=json.loads((root/'plan.json').read_text())
    assert digest(Path(__file__))==plan['source_sha256']
    assert digest(Path(baseline.__file__))==plan['baseline_source_sha256']
    assert digest(Path(baseline.__file__).with_name('environment.py'))==plan['environment_source_sha256']
    assert digest(Path(__file__).with_name('verify_run.py'))==plan['verifier_source_sha256']
    assert digest(Path(plan['dataset']))==plan['dataset_sha256']
    assert hashlib.sha256(SYSTEM.encode()).hexdigest()==plan['system_prompt_sha256']
    assert prefix_hash()==plan['demonstrations_sha256']
    return plan


def run_one(root,info,seed):
    plan=validate(root)
    model_info=json.loads(info.read_text())
    assert model_info['model_id']==plan['model_id'] and model_info['revision']==plan['model_revision']
    if seed not in plan['seeds']:
        raise ValueError('unplanned seed')
    out=root/f'native_examples-seed{seed}'
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())['status']=='complete':
        return
    original_episode=baseline.run_episode
    def episode(task,generate,*args,**kwargs):
        inserted=False
        def generate_with_examples(messages,request_seed):
            nonlocal inserted
            if not inserted:
                assert len(messages)==2
                messages[1:1]=deepcopy(DEMOS)
                inserted=True
            return generate(messages,request_seed)
        result=original_episode(task,generate_with_examples,*args,**kwargs)
        result['question_index']=len(DEMOS)+1
        result['demonstrations_sha256']=prefix_hash()
        return result
    baseline.run_episode=episode
    baseline.SYSTEM_PROMPT=SYSTEM
    sys.argv=['toolrl.baseline','--data',plan['dataset'],'--model-info',str(info),
              '--out',str(out),'--parser','fenced','--seed',str(seed)]
    if (out/'episodes.jsonl').exists():
        sys.argv.append('--resume')
    baseline.main()
    manifest=json.loads((out/'manifest.json').read_text())
    manifest['chat_prompt_probe']=dict(plan_sha256=digest(root/'plan.json'),source_sha256=plan['source_sha256'],
                                      demonstrations_sha256=prefix_hash(),question_index=len(DEMOS)+1)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')


def run_all(root,info):
    plan=validate(root)
    for seed in plan['seeds']:
        print(json.dumps({'event':'native_examples_started','seed':seed}),flush=True)
        subprocess.run([sys.executable,'-m','toolrl.chat_prompt_probe','run-one','--root',str(root),
                        '--model-info',str(info),'--seed',str(seed)],check=True)
        out=root/f'native_examples-seed{seed}'
        subprocess.run([sys.executable,'-m','toolrl.analyze',str(out)],check=True,capture_output=True,text=True)
        subprocess.run([sys.executable,'-m','toolrl.verify_run',str(out),'--data',plan['dataset'],
                        '--model-info',str(info)],check=True,capture_output=True,text=True)
        m=json.loads((out/'summary.json').read_text())
        print(json.dumps({'event':'native_examples_complete','seed':seed,'correct':m['overall']['successes'],
                          'by_kind':{k:v['successes'] for k,v in m['by_kind'].items()}}),flush=True)


def analyze(root):
    plan=validate(root)
    tasks={t['task_id']:t for t in [json.loads(x) for x in Path(plan['dataset']).read_text().splitlines()]}
    combined=[]
    conditions=[]
    for seed in plan['seeds']:
        path=root/f'native_examples-seed{seed}'
        manifest=json.loads((path/'manifest.json').read_text())
        verification=json.loads((path/'verification.json').read_text())
        assert manifest['status']=='complete' and verification['episodes_sha256']==digest(path/'episodes.jsonl')
        config=manifest['config']
        expected=dict(model_id=plan['model_id'],model_revision=plan['model_revision'],
                      dataset_sha256=plan['dataset_sha256'],seed=seed,task_count=60,
                      baseline_source_sha256=plan['baseline_source_sha256'],
                      environment_source_sha256=plan['environment_source_sha256'],
                      system_prompt_sha256=plan['system_prompt_sha256'],parser='fenced',enable_thinking=False,
                      do_sample=True,temperature=.7,top_p=.8,top_k=20,max_context=1024,max_new_tokens=64,
                      max_calls=3,max_turns=4,dtype='bfloat16',device='cuda:0',batch_size=1,cpu_threads=4)
        assert all(config.get(key)==value for key,value in expected.items())
        rows=[json.loads(x) for x in (path/'episodes.jsonl').read_text().splitlines()]
        assert {r['task_id'] for r in rows}==tasks.keys()
        for row in rows:
            assert row['question_index']==len(DEMOS)+1 and row['messages'][1:row['question_index']]==DEMOS
            assert row['demonstrations_sha256']==plan['demonstrations_sha256']
            assert hashlib.sha256(row['messages'][0]['content'].encode()).hexdigest()==plan['system_prompt_sha256']
        combined.extend(rows)
        conditions.append(dict(seed=seed,metrics=diagnostics(rows,tasks),verification=verification,
                               inference_wall_seconds=manifest['inference_wall_seconds'],
                               peak_cuda_allocated_mib=manifest['cuda_peak_allocated_mib']))
    pairs={}
    for variant in ['original','neutral','balanced_examples']:
        prior=[]
        for seed in plan['seeds']:
            prior.extend(json.loads(x) for x in (Path(plan['primary_root'])/f'{variant}-seed{seed}'/'episodes.jsonl').read_text().splitlines())
        pairs[f'{variant}_to_native_examples']=paired(prior,combined)
    summary=dict(plan=plan,aggregate=diagnostics(combined,tasks),conditions=conditions,paired=pairs,
                 verified_requests=sum(c['verification']['requests_verified'] for c in conditions))
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['aggregate','paired','verified_requests']},indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','run-all','run-one','analyze'])
    parser.add_argument('--root',type=Path,default=Path('results/native-examples-v1'))
    parser.add_argument('--primary',type=Path,default=Path('results/prompt-control-v1'))
    parser.add_argument('--model-info',type=Path)
    parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.command=='prepare':
        prepare(args.root,args.primary,args.model_info)
    elif args.command=='run-all':
        run_all(args.root,args.model_info)
    elif args.command=='run-one':
        run_one(args.root,args.model_info,args.seed)
    else:
        analyze(args.root)


if __name__=='__main__':
    main()
