import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'outputs/budgeted-tool-rl/results/sft-v3'
result=[]
for d in sorted((root/'eval').glob('*')):
    m=json.loads((d/'manifest.json').read_text(encoding='utf-8'))
    rs=[json.loads(x) for x in (d/'episodes.jsonl').read_text(encoding='utf-8').splitlines()] if (d/'episodes.jsonl').exists() else []
    result.append({'run':d.name,'status':m['status'],'completed':len(rs),'total':m['config']['task_count'],'correct':sum(r['success'] for r in rs),
                   'extra_calls_on_success':sum(r['correct_extra_calls'] or 0 for r in rs),
                   'first_project_employee':sum(bool(r['actions']) and r['actions'][0].get('field')=='employee' and r['actions'][0].get('entity','').startswith('project_') for r in rs)})
print(json.dumps({'complete_cohorts':sum(r['status']=='complete' for r in result),'episodes':sum(r['completed'] for r in result),
                  'active':[r for r in result if r['status']!='complete'],'last_complete':[r for r in result if r['status']=='complete'][-3:]}))
