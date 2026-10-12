"""Preregistered paired prompt controls using the unchanged baseline engine.

Only the engine's process-local SYSTEM_PROMPT binding is substituted. Every
request still uses the original model loading, generation, parsing and tools.
The wrapper and prompt hashes are recorded separately for reproducibility.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from . import baseline
from .dataset import make_tasks, write_tasks
from .environment import SYSTEM_PROMPT

PATH_SENTENCE = "An owner is an employee ID. To find the owner's department, first look up the project's owner, then look up that employee's department."
NEUTRAL = SYSTEM_PROMPT.replace(PATH_SENTENCE, 'An owner is an employee ID.')
EXAMPLES = '''
Examples of complete independent episodes follow. Their entities and values
are illustrative, not facts for the current question. Use the current question
to select the requested field, and use actual tool results for private values.

Example 1 (the budget is already supplied):
User: The budget of project_demo_417f is 321. What is its budget?
Assistant: {"action":"answer","value":321}

Example 2 (a private budget):
User: What is the budget of project_demo_62c8?
Assistant: {"action":"lookup","entity":"project_demo_62c8","field":"budget"}
Tool: {"entity":"project_demo_62c8","field":"budget","value":875}
Assistant: {"action":"answer","value":875}

Example 3 (a private owner's department):
User: What department does the owner of project_demo_19a4 belong to?
Assistant: {"action":"lookup","entity":"project_demo_19a4","field":"owner"}
Tool: {"entity":"project_demo_19a4","field":"owner","value":"employee_demo_8bd9"}
Assistant: {"action":"lookup","entity":"employee_demo_8bd9","field":"department"}
Tool: {"entity":"employee_demo_8bd9","field":"department","value":"department_48216"}
Assistant: {"action":"answer","value":"department_48216"}

The examples have ended. Now solve the current user question, emitting only
the next single JSON action. Never copy an example entity into the current episode.
'''
PROMPTS = {'original': SYSTEM_PROMPT, 'neutral': NEUTRAL, 'balanced_examples': NEUTRAL + EXAMPLES}
SEEDS = [7, 10007, 20007]
ORDER = [['original', 'neutral', 'balanced_examples'],
         ['neutral', 'balanced_examples', 'original'],
         ['balanced_examples', 'original', 'neutral']]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root, data):
    if (root / 'plan.json').exists() or data.exists():
        raise ValueError('study already prepared; existing plan/data must not be overwritten')
    tasks = make_tasks(884201, (20, 20, 20), 'p01')
    old = Path('data/dev100.jsonl')
    if old.exists():
        prior = [json.loads(line) for line in old.read_text(encoding='utf-8').splitlines()]
        assert not ({e for t in tasks for e in t['database']} & {e for t in prior for e in t['database']})
    root.mkdir(parents=True, exist_ok=True)
    (root / 'prompts').mkdir()
    write_tasks(data, tasks, 884201)
    for variant, prompt in PROMPTS.items():
        (root / 'prompts' / (variant + '.txt')).write_text(prompt, encoding='utf-8', newline='\n')
    plan = dict(purpose='paired prompt diagnostic, not final generalization test',
                prepared_at=baseline.datetime.now(baseline.timezone.utc).isoformat(),
                tasks=60, tasks_per_family=20, repeats_per_task=3,
                total_episodes=540, dataset=str(data), dataset_sha256=digest(data),
                seeds=SEEDS, order=ORDER, prompt_sha256={v: digest(root/'prompts'/(v+'.txt')) for v in PROMPTS},
                baseline_source_sha256=digest(Path(baseline.__file__)),
                environment_source_sha256=digest(Path(baseline.__file__).with_name('environment.py')),
                study_source_sha256=digest(Path(__file__)),
                fixed=dict(model='Qwen/Qwen3-0.6B', model_revision='c1899de289a04d12100db370d81485cdf75e47ca',
                           parser='fenced', thinking=False, temperature=.7, top_p=.8, top_k=20,
                           max_context=1024, max_new_tokens=64, max_calls=3, max_turns=4),
                comparisons=['original vs neutral isolates removal of the specific owner-department plan sentence',
                             'neutral vs balanced_examples adds three complete demonstrations; length and demonstration effects are bundled'],
                metrics=['success and strict success by family and seed', 'correct first lookup',
                         'two-lookup tasks starting lookup', 'budget tasks selecting owner',
                         'paired successes gained/lost', 'raw JSON validity', 'context/truncation failures'],
                interpretation='Improvement is evidence of prompt sensitivity. Continued failure cannot prove a model capability ceiling. Repeated seeds share tasks; trials are not independent benchmark items.')
    (root/'plan.json').write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'event':'prepared', 'tasks':60, 'planned_real_episodes':540}), flush=True)


def validate_plan(root):
    plan=json.loads((root/'plan.json').read_text(encoding='utf-8'))
    paths = {'dataset_sha256': Path(plan['dataset']), 'study_source_sha256': Path(__file__),
             'baseline_source_sha256': Path(baseline.__file__),
             'environment_source_sha256': Path(baseline.__file__).with_name('environment.py')}
    for key, path in paths.items():
        if digest(path)!=plan[key]:
            raise ValueError('preregistered provenance changed: '+key)
    for variant, sha in plan['prompt_sha256'].items():
        if digest(root/'prompts'/(variant+'.txt'))!=sha:
            raise ValueError('prompt changed: '+variant)
    return plan


def run_one(root, info, variant, seed):
    plan=validate_plan(root)
    if seed not in plan['seeds'] or variant not in plan['prompt_sha256']:
        raise ValueError('unplanned condition')
    model_info=json.loads(info.read_text(encoding='utf-8'))
    if model_info['model_id']!=plan['fixed']['model'] or model_info['revision']!=plan['fixed']['model_revision']:
        raise ValueError('unplanned model')
    out=root/f'{variant}-seed{seed}'
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text(encoding='utf-8'))['status']=='complete':
        print(json.dumps({'event':'already_complete', 'variant':variant,'seed':seed}),flush=True)
        return
    prompt=(root/'prompts'/(variant+'.txt')).read_text(encoding='utf-8')
    previous_prompt, previous_args=baseline.SYSTEM_PROMPT, sys.argv
    baseline.SYSTEM_PROMPT=prompt
    sys.argv=['toolrl.baseline','--data',plan['dataset'],'--model-info',str(info),
              '--out',str(out),'--parser','fenced','--seed',str(seed)]
    if (out/'episodes.jsonl').exists():
        sys.argv.append('--resume')
    try:
        baseline.main()
    finally:
        baseline.SYSTEM_PROMPT, sys.argv=previous_prompt, previous_args
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    manifest['prompt_study']=dict(variant=variant,seed=seed,plan_sha256=digest(root/'plan.json'),
                                  study_source_sha256=plan['study_source_sha256'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')


def run_all(root, info):
    plan=validate_plan(root)
    for seed, variants in zip(plan['seeds'],plan['order']):
        for variant in variants:
            print(json.dumps({'event':'condition_started','variant':variant,'seed':seed}),flush=True)
            subprocess.run([sys.executable,'-m','toolrl.prompt_study','run-one','--root',str(root),
                            '--model-info',str(info),'--variant',variant,'--seed',str(seed)],check=True)
            out=root/f'{variant}-seed{seed}'
            # Do not flood the progress stream with full numerical summaries.
            subprocess.run([sys.executable,'-m','toolrl.analyze',str(out)],check=True,capture_output=True,text=True)
            subprocess.run([sys.executable,'-m','toolrl.verify_run',str(out),'--data',plan['dataset'],
                            '--model-info',str(info)],check=True,capture_output=True,text=True)
            summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
            print(json.dumps({'event':'condition_complete','variant':variant,'seed':seed,
                              'correct':summary['overall']['successes'],
                              'by_kind':{k:v['successes'] for k,v in summary['by_kind'].items()}}),flush=True)


def diagnostics(rows, tasks):
    def first_lookup(row):
        return next((a for a in row['actions'] if a['action']=='lookup'),None)
    eligible=[r for r in rows if r['minimum_calls']>0]
    two=[r for r in rows if r['kind']=='two_lookup']
    one=[r for r in rows if r['kind']=='one_lookup']
    correct_first=sum(first_lookup(r)=={'action':'lookup',**tasks[r['task_id']]['oracle_calls'][0]} for r in eligible)
    requests=sum(len(r['requests']) for r in rows)
    return dict(episodes=len(rows),successes=sum(r['success'] for r in rows),
                strict_successes=sum(r['strict_success'] for r in rows),
                requests=requests,strict_valid_requests=sum(r['strict_valid_requests'] for r in rows),
                strict_valid_request_rate=sum(r['strict_valid_requests'] for r in rows)/requests,
                correct_first_lookup=correct_first,private_episodes=len(eligible),
                two_lookup_started=sum(r['calls']>0 for r in two),two_lookup_attempts=len(two),
                budget_first_lookup_owner=sum((first_lookup(r) or {}).get('field')=='owner' for r in one),
                one_lookup_attempts=len(one),
                stops=dict(Counter(r['stop_reason'] for r in rows)),
                by_kind={k:dict(attempts=sum(r['kind']==k for r in rows),
                                successes=sum(r['success'] for r in rows if r['kind']==k),
                                strict_successes=sum(r['strict_success'] for r in rows if r['kind']==k))
                         for k in ('direct','one_lookup','two_lookup')},
                by_template={f'{k}:{i}':dict(attempts=sum(r['kind']==k and r['template_id']==i for r in rows),
                                             successes=sum(r['success'] for r in rows if r['kind']==k and r['template_id']==i))
                             for k in ('direct','one_lookup','two_lookup') for i in (0,1,2)})


def paired(left,right):
    a={(r['task_id'],r['seed']):r for r in left}
    b={(r['task_id'],r['seed']):r for r in right}
    if a.keys()!=b.keys():
        raise ValueError('unpaired task/seed outcomes')
    return dict(paired_attempts=len(a),both_correct=sum(a[k]['success'] and b[k]['success'] for k in a),
                gained=sum(not a[k]['success'] and b[k]['success'] for k in a),
                lost=sum(a[k]['success'] and not b[k]['success'] for k in a),
                both_wrong=sum(not a[k]['success'] and not b[k]['success'] for k in a))


def analyze(root):
    plan=validate_plan(root)
    tasks={t['task_id']:t for t in [json.loads(x) for x in Path(plan['dataset']).read_text(encoding='utf-8').splitlines()]}
    rows_by_variant, conditions={},[]
    for variant in plan['prompt_sha256']:
        combined=[]
        for seed in plan['seeds']:
            path=root/f'{variant}-seed{seed}'
            manifest=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
            verification=json.loads((path/'verification.json').read_text(encoding='utf-8'))
            if manifest['status']!='complete' or manifest['config']['system_prompt_sha256']!=plan['prompt_sha256'][variant]:
                raise ValueError('incomplete/mismatched condition')
            config=manifest['config']
            expected_config=dict(model_id=plan['fixed']['model'],model_revision=plan['fixed']['model_revision'],
                                 dataset_sha256=plan['dataset_sha256'],seed=seed,task_count=plan['tasks'],
                                 baseline_source_sha256=plan['baseline_source_sha256'],
                                 environment_source_sha256=plan['environment_source_sha256'],
                                 parser='fenced',enable_thinking=False,do_sample=True,temperature=.7,top_p=.8,top_k=20,
                                 max_context=1024,max_new_tokens=64,max_calls=3,max_turns=4,
                                 dtype='bfloat16',device='cuda:0',batch_size=1,cpu_threads=4)
            if any(config.get(key)!=value for key,value in expected_config.items()):
                raise ValueError('non-prompt configuration changed')
            if digest(path/'episodes.jsonl')!=verification['episodes_sha256']:
                raise ValueError('verified traces changed')
            rows=[json.loads(x) for x in (path/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
            if {r['task_id'] for r in rows}!=tasks.keys():
                raise ValueError('task set changed')
            for row in rows:
                if hashlib.sha256(row['messages'][0]['content'].encode()).hexdigest()!=plan['prompt_sha256'][variant]:
                    raise ValueError('model-visible prompt hash mismatch')
            combined.extend(rows)
            conditions.append(dict(variant=variant,seed=seed,metrics=diagnostics(rows,tasks),
                                   inference_wall_seconds=manifest['inference_wall_seconds'],
                                   peak_cuda_allocated_mib=manifest['cuda_peak_allocated_mib'],
                                   verification=verification))
        rows_by_variant[variant]=combined
    summary=dict(plan=plan,conditions=conditions,
                 aggregate={v:diagnostics(rows,tasks) for v,rows in rows_by_variant.items()},
                 paired={f'{a}_to_{b}':paired(rows_by_variant[a],rows_by_variant[b])
                         for a,b in [('original','neutral'),('neutral','balanced_examples'),('original','balanced_examples')]},
                 unique_tasks=len(tasks),real_episodes=sum(len(rows) for rows in rows_by_variant.values()),
                 verified_requests=sum(c['verification']['requests_verified'] for c in conditions),
                 note='180 attempts per prompt = 60 shared tasks x 3 seeds, not 180 independent tasks; no training performed.')
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('aggregate','paired','real_episodes','verified_requests')},indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','run-all','run-one','analyze'])
    parser.add_argument('--root',type=Path,default=Path('results/prompt-control-v1'))
    parser.add_argument('--data',type=Path,default=Path('data/prompt_control60.jsonl'))
    parser.add_argument('--model-info',type=Path)
    parser.add_argument('--variant',choices=list(PROMPTS))
    parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.command=='prepare':
        prepare(args.root,args.data)
    elif args.command=='run-all':
        run_all(args.root,args.model_info)
    elif args.command=='run-one':
        run_one(args.root,args.model_info,args.variant,args.seed)
    else:
        analyze(args.root)


if __name__=='__main__':
    main()
