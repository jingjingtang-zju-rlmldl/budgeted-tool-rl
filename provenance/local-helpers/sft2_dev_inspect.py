from pathlib import Path
import json,sys,os
project=Path(__file__).resolve().parents[1]/'outputs/budgeted-tool-rl'
sys.path.insert(0,str(project));os.chdir(project)
from toolrl.sft2_report import metrics
from toolrl.sft2_experiment import rows,CONDITIONS
root=project/'results/sft-v2'
for model,prompt in CONDITIONS:
    rs=rows(root/'eval'/f'dev-{model}-{prompt}-seed7'/'episodes.jsonl')
    print(json.dumps({'condition':model+'/'+prompt,**metrics(rs)}))
    if model=='old' and prompt=='protocol':
        for r in rs:
            if not r['success']:
                print(json.dumps({k:r[k] for k in ('task_id','kind','stop_reason','calls','answer','expected')}
                    |{'outputs':[q['text'] for q in r['requests']]}))
