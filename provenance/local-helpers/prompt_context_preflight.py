import json
import sys
from pathlib import Path

root=Path(__file__).resolve().parent.parent
project=root/'outputs'/'budgeted-tool-rl'
sys.path.insert(0,str(project))
from transformers import AutoTokenizer
from toolrl.prompt_study import PROMPTS
from toolrl.environment import LookupEnvironment

info=json.loads((root/'work'/'model-location.json').read_text())
tokenizer=AutoTokenizer.from_pretrained(info['snapshot'],local_files_only=True)
tasks=[json.loads(x) for x in (project/'data'/'prompt_control60.jsonl').read_text().splitlines()]
sizes={}
for variant,prompt in PROMPTS.items():
    lengths=[]
    for task in tasks:
        env=LookupEnvironment(task)
        messages=[dict(role='system',content=prompt),dict(role='user',content=task['prompt'])]
        lengths.append(len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)))
        for call in task['oracle_calls']:
            messages.append(dict(role='assistant',content=json.dumps(dict(action='lookup',**call))))
            messages.append(dict(role='tool',content=json.dumps(env.lookup(call['entity'],call['field']),sort_keys=True)))
            lengths.append(len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)))
    sizes[variant]=dict(min_tokens=min(lengths),max_oracle_context_tokens=max(lengths))
assert max(v['max_oracle_context_tokens'] for v in sizes.values())<1024
(project/'results'/'prompt-control-v1'/'context_preflight.json').write_text(json.dumps(sizes,indent=2))
print(json.dumps(sizes,indent=2))
