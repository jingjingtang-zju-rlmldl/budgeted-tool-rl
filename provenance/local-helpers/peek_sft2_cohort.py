from pathlib import Path
import argparse,json,sys,os
project=Path(__file__).resolve().parents[1]/'outputs/budgeted-tool-rl'
sys.path.insert(0,str(project));os.chdir(project)
from toolrl.sft2_report import metrics
from toolrl.sft2_experiment import rows
p=argparse.ArgumentParser();p.add_argument('cohort');a=p.parse_args()
rs=rows(project/'results/sft-v2/eval'/a.cohort/'episodes.jsonl')
print(json.dumps({'partial':True,'cohort':a.cohort,**metrics(rs)}))
for r in rs:
    if not r['success']:
        print(json.dumps({k:r[k] for k in ('task_id','kind','seed','answer','expected','calls','minimum_calls','stop_reason')}
            |{'outputs':[q['text'] for q in r['requests']]}))
