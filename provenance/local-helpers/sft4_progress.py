import json
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[1]/'outputs/budgeted-tool-rl/results/sft-v4'
sys.path.insert(0,str(root.parents[1]))
from toolrl.sft4_data import harvest
done=[];active=[];total=0
natural_facts=set();natural_types={}
paths=list((root/'eval').glob('*/manifest.json'))+list((root/'copy-probe/eval').glob('*/manifest.json'))
for path in sorted(paths,key=lambda p:p.stat().st_mtime):
 m=json.loads(path.read_text(encoding='utf-8'));p=path.parent/'episodes.jsonl'
 rs=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()] if p.exists() else []
 total+=len(rs)
 r=dict(run=path.parent.name,status=m['status'],completed=len(rs),total=m['config']['task_count'],
        grounded_correct=sum(x['grounded_success'] for x in rs),exact_correct=sum(x['success'] for x in rs),
        extra_cost_on_success=sum(x['correct_extra_cost'] or 0 for x in rs),
        unknown_entities=sum(o.get('error')=='unknown_entity' for x in rs for o in x['observations']))
 if m['config']['split'] in ('collect','reserve','copy_probe'):
  dataset=(root/'copy-probe/data/copy_probe.jsonl' if m['config']['split']=='copy_probe' else root.parents[1]/'data/sft-v4'/(m['config']['split']+'.jsonl'))
  tasks={t['task_id']:t for t in map(json.loads,dataset.read_text(encoding='utf-8').splitlines())}
  for row in rs:
   t=harvest(tasks[row['task_id']],row)
   if t and t['fact_group'] not in natural_facts:
    natural_facts.add(t['fact_group']);natural_types[t['natural_copy_type']]=natural_types.get(t['natural_copy_type'],0)+1
 (done if m['status']=='complete' else active).append(r)
out=dict(complete_cohorts=len(done),actual_episodes=total,active=active,natural_copy_facts=len(natural_facts),natural_copy_types=natural_types,last_complete=done[-1:])
for name in ['training-set-manifest','memory-probe/manifest','training/manifest']:
 path=root/(name+'.json')
 if path.exists():
  m=json.loads(path.read_text(encoding='utf-8'));out[name]={k:m[k] for k in ['status','natural_training_tasks','training_tasks','decision_targets','optimizer_steps','microbatches_seen','cuda_peak_allocated_mib'] if k in m}
print(json.dumps(out))
