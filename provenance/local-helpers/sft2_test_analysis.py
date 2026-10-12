"""Analyze every paired test before opening the sealed signal stage."""
import json,sys
from pathlib import Path
from datetime import datetime,timezone
project=Path(__file__).resolve().parents[1]/'outputs/budgeted-tool-rl'
sys.path.insert(0,str(project))
from toolrl.sft2_experiment import CONDITIONS,read,rows,sha256,validate,checkpoint
from toolrl.sft2_report import grouped,paired

root=project/'results/sft-v2'
import os
os.chdir(project)
plan=validate(root/'plan.json');checkpoint(plan,root/'plan.json')
data={}
for model,prompt in CONDITIONS:
    combined=[]
    for seed in plan['evaluation_seeds']['test']:
        cohort=root/'eval'/f'test-{model}-{prompt}-seed{seed}'
        m=read(cohort/'manifest.json');v=read(cohort/'verification.json')
        assert m['status']=='complete' and v['tasks_verified']==72
        assert sha256(cohort/'episodes.jsonl')==v['episodes_sha256']
        combined+=rows(cohort/'episodes.jsonl')
    data[model+'/'+prompt]=combined
assert not any(root.glob('eval/signal-*/episodes.jsonl'))
result=dict(analyzed_at=datetime.now(timezone.utc).isoformat(),
    plan_sha256=sha256(root/'plan.json'),signal_not_opened=True,
    metrics={k:grouped(rs) for k,rs in data.items()},
    comparisons={k:paired(rs,data['new/protocol']) for k,rs in data.items() if k!='new/protocol'})
(root/'test-analysis-before-signal.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
fields=('attempts','successes','strict_successes','grounded_successes','correct_extra_calls','requests',
        'failed_stops','observed_error_episodes','correct_after_error','necessary_retries','followed_redirects',
        'truncated_requests','simulated_tool_requests','multiple_action_marker_requests')
print(json.dumps({'metrics':{key:{field:m[field] for field in fields}|{'by_kind':{k:v['successes'] for k,v in m['by_kind'].items()}} for key,m in result['metrics'].items()},
                  'comparisons':result['comparisons']},indent=2))
print('NEW MODEL FAILURES:')
for r in data['new/protocol']:
    if not r['success']:
        print(json.dumps({k:r[k] for k in ('task_id','kind','seed','answer','expected','calls','minimum_calls','stop_reason','actions','observations')}))
